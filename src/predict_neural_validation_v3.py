"""Generate evidence-bound neural-v3 predictions without target-label access."""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.special import expit
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from prepare_feature_cache import vectorizer
from train_neural_validation_v3 import (CodeBertDetector, TokenCache, atomic_json,
                                         log, predict, runtime, set_seed)
from vultriage.data import sha256

ROOT=Path(__file__).resolve().parents[1]


def relative(path: Path):
    resolved=path.resolve()
    if not resolved.is_relative_to(ROOT):
        raise ValueError("Prediction evidence path escapes workspace")
    return resolved.relative_to(ROOT).as_posix()


def read_rows(path: Path,source: bool):
    rows=[json.loads(line) for line in path.open(encoding="utf-8")]
    if [row["position"] for row in rows]!=list(range(len(rows))):
        raise ValueError("Record positions not contiguous")
    if not source and any("label" in row or "target" in row for row in rows):
        raise ValueError("Predictor received target labels")
    return rows


def role_positions(rows,role):
    return np.asarray([i for i,row in enumerate(rows) if row["role"]==role],dtype=np.int64)


def atomic_npz(path: Path,**arrays):
    temporary=path.with_suffix(path.suffix+".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream,**arrays)
    temporary.replace(path)


def hashed_features(source_path,target_path,output):
    output.mkdir(parents=True,exist_ok=True)
    manifest=output/"manifest.json"
    expected=dict(source_rows_sha256=sha256(source_path),target_rows_sha256=sha256(target_path))
    if manifest.exists():
        recorded=json.loads(manifest.read_text(encoding="utf-8"))
        if any(recorded[k]!=v for k,v in expected.items()):
            raise ValueError("Hashing cache input mismatch")
        for name,digest in recorded["hashes"].items():
            if sha256(output/name)!=digest:
                raise ValueError("Hashing feature cache digest mismatch")
        return sparse.load_npz(output/"source_features.npz"),sparse.load_npz(output/"target_features.npz")
    transform=vectorizer(json.loads((ROOT/"configs/preregistered_experiment.json").read_text(encoding="utf-8")))
    matrices=[]
    for path,source,name in ((source_path,True,"source_features.npz"),(target_path,False,"target_features.npz")):
        rows=read_rows(path,source)
        blocks=[]
        for begin in range(0,len(rows),512):
            blocks.append(transform.transform([r["text"] for r in rows[begin:begin+512]]).astype(np.float32))
        matrix=sparse.vstack(blocks,format="csr") if blocks else sparse.csr_matrix((0,262144),dtype=np.float32)
        sparse.save_npz(output/name,matrix)
        matrices.append(matrix)
        log(f"Built {name}: {matrix.shape}, nnz={matrix.nnz}")
    atomic_json(manifest,dict(**expected,hashes={name:sha256(output/name) for name in ("source_features.npz","target_features.npz")}))
    return tuple(matrices)


def fit_hashing(args,source_rows,source_features,cfg):
    train=role_positions(source_rows,"train")
    valid=role_positions(source_rows,"model_validation")
    labels=np.asarray([r["label"] for r in source_rows],dtype=np.int64)
    fits={}
    args.training.mkdir(parents=True,exist_ok=True)
    with threadpool_limits(limits=4):
        for seed in cfg["fine_tuned_codebert"]["seeds"]:
            folder=args.training/f"seed-{seed}"
            folder.mkdir(parents=True,exist_ok=True)
            receipt=folder/"training_manifest.json"
            checkpoint=folder/"detector.npz"
            if receipt.exists():
                old=json.loads(receipt.read_text(encoding="utf-8"))
                if old["config_sha256"]!=sha256(args.config) or sha256(checkpoint)!=old["checkpoint_sha256"]:
                    raise ValueError("Hashing detector seal changed")
                fitted=np.load(checkpoint,allow_pickle=False)
                fits[seed]=(fitted["coef"],fitted["intercept"],receipt,checkpoint)
                continue
            started=time.perf_counter()
            best=-float("inf")
            best_alpha=None
            selected=None
            candidates=[]
            for alpha in (1e-6,1e-5,1e-4):
                model=SGDClassifier(loss="log_loss",penalty="l2",alpha=alpha,max_iter=3,tol=None,
                                    average=True,random_state=seed,class_weight=None)
                model.fit(source_features[train],labels[train])
                p=model.predict_proba(source_features[valid])[:,1]
                ap=float(average_precision_score(labels[valid],p))
                candidates.append(dict(alpha=alpha,source_validation_ap=ap))
                if ap>best or (ap==best and (best_alpha is None or alpha>best_alpha)):
                    best,best_alpha,selected=ap,alpha,model
            atomic_npz(checkpoint,coef=selected.coef_,intercept=selected.intercept_)
            atomic_json(receipt,dict(detector="hashing_sgd",seed=seed,actual_independent_fit=True,
                config_sha256=sha256(args.config),source_rows_sha256=sha256(args.source_rows),
                checkpoint_sha256=sha256(checkpoint),selected_alpha=best_alpha,source_validation_ap=best,
                candidates=candidates,target_labels_accessed=False,target_inputs_used_for_selection=False,
                source_validation_refit=False,script_sha256=sha256(Path(__file__)),wall_seconds=time.perf_counter()-started))
            fits[seed]=(selected.coef_,selected.intercept_,receipt,checkpoint)
            log(f"Hashing actual fit {seed}: source AP={best:.6f}, alpha={best_alpha}")
    return fits


def base_embeddings(args,cache_path: Path,output: Path):
    manifest=output.with_suffix(".manifest.json")
    if manifest.exists():
        record=json.loads(manifest.read_text(encoding="utf-8"))
        if record["token_manifest_sha256"]!=sha256(cache_path/"tokenization_manifest.json") or record["embedding_sha256"]!=sha256(output):
            raise ValueError("Base embedding cache changed")
        return np.load(output,mmap_mode="r")
    import torch
    cache=TokenCache(cache_path,source=cache_path==args.source_tokens)
    set_seed(13)
    model=CodeBertDetector(args.model_path).cuda()
    started=time.perf_counter()
    _,features=predict(model,cache,np.arange(len(cache.rows)),embeddings=True)
    output.parent.mkdir(parents=True,exist_ok=True)
    np.save(output,features)
    atomic_json(manifest,dict(token_manifest_sha256=sha256(cache_path/"tokenization_manifest.json"),
        embedding_sha256=sha256(output),encoder_revision="3b0952feddeffad0063f274080e3c23d75e7eb39",
        target_labels_accessed=False,wall_seconds=time.perf_counter()-started,runtime=runtime()))
    del model
    torch.cuda.empty_cache()
    return features


def fit_frozen(args,rows,features):
    args.training.mkdir(parents=True,exist_ok=True)
    checkpoint=args.training/"detector.npz"
    receipt=args.training/"training_manifest.json"
    if receipt.exists():
        old=json.loads(receipt.read_text(encoding="utf-8"))
        if old["config_sha256"]!=sha256(args.config) or old["checkpoint_sha256"]!=sha256(checkpoint):
            raise ValueError("Frozen-head seal changed")
        return np.load(checkpoint,allow_pickle=False),receipt,checkpoint
    started=time.perf_counter()
    train,valid=(role_positions(rows,r) for r in ("train","model_validation"))
    y=np.asarray([r["label"] for r in rows],dtype=np.int64)
    scaler=StandardScaler()
    train_x=scaler.fit_transform(features[train])
    valid_x=scaler.transform(features[valid])
    best=-float("inf")
    selected=None
    best_c=None
    candidates=[]
    with threadpool_limits(limits=4):
        for c in (.01,.1,1.,10.):
            model=LogisticRegression(C=c,solver="liblinear",penalty="l2",class_weight=None,
                                     max_iter=2000,tol=1e-6,random_state=13)
            model.fit(train_x,y[train])
            ap=float(average_precision_score(y[valid],model.predict_proba(valid_x)[:,1]))
            candidates.append(dict(C=c,source_validation_ap=ap))
            log(f"Frozen head C={c}: source AP={ap:.6f}")
            if ap>best:
                best,best_c,selected=ap,c,model
    arrays=dict(coef=selected.coef_,intercept=selected.intercept_,mean=scaler.mean_,scale=scaler.scale_)
    atomic_npz(checkpoint,**arrays)
    atomic_json(receipt,dict(detector="frozen_codebert",seed=13,actual_independent_neural_fit=False,
        actual_head_fit_count=1,encoder_updated=False,source_validation_refit=False,
        config_sha256=sha256(args.config),source_rows_sha256=sha256(args.source_rows),
        checkpoint_sha256=sha256(checkpoint),selected_C=best_c,source_validation_ap=best,
        candidates=candidates,target_labels_accessed=False,target_inputs_used_for_selection=False,
        script_sha256=sha256(Path(__file__)),wall_seconds=time.perf_counter()-started))
    return arrays,receipt,checkpoint


def write_prediction(args,seed,source_positions,source_p,target_p,source_features,target_features,
                     receipt,checkpoint,feature_cache=None):
    folder=args.output/f"seed-{seed}"
    folder.mkdir(parents=True,exist_ok=True)
    npz=folder/"predictions.npz"
    metadata=folder/"metadata.json"
    if metadata.exists():
        old=json.loads(metadata.read_text(encoding="utf-8"))
        if old["prediction_sha256"]!=sha256(npz):
            raise ValueError("Completed predictions changed")
        log(f"Verified existing {args.detector}/{args.dataset}/{seed}")
        return
    arrays=dict(source_positions=source_positions,source_probability=np.asarray(source_p,dtype=np.float32),
                target_probability=np.asarray(target_p,dtype=np.float32),target_positions=np.arange(len(target_p),dtype=np.int64))
    if source_features is not None:
        arrays.update(source_features=np.asarray(source_features,dtype=np.float32),target_features=np.asarray(target_features,dtype=np.float32))
    if not np.isfinite(source_p).all() or not np.isfinite(target_p).all():
        raise ValueError("Nonfinite detector probabilities")
    atomic_npz(npz,**arrays)
    sources=dict(training_manifest_path=relative(receipt),training_manifest_sha256=sha256(receipt),
                 checkpoint_path=relative(checkpoint),checkpoint_sha256=sha256(checkpoint))
    if args.source_tokens is not None:
        sources.update(source_token_manifest_path=relative(args.source_tokens/"tokenization_manifest.json"),
                       source_token_manifest_sha256=sha256(args.source_tokens/"tokenization_manifest.json"),
                       target_token_manifest_path=relative(args.target_tokens/"tokenization_manifest.json"),
                       target_token_manifest_sha256=sha256(args.target_tokens/"tokenization_manifest.json"))
    if feature_cache is not None:
        sources.update(feature_cache_path=relative(feature_cache),feature_cache_manifest_sha256=sha256(feature_cache/"manifest.json"))
    atomic_json(metadata,dict(detector=args.detector,dataset=args.dataset,seed=seed,
        prediction_sha256=sha256(npz),prediction_path=relative(npz),config_sha256=sha256(args.config),
        dataset_config_sha256=sha256(args.dataset_config) if args.dataset_config else None,
        source_rows_path=relative(args.source_rows),source_rows_sha256=sha256(args.source_rows),
        target_rows_path=relative(args.target_rows),target_rows_sha256=sha256(args.target_rows),
        target_labels_accessed=False,target_labels_used_for_selection=False,
        probability_storage="float32",source_roles_predicted=["model_validation","calibration"],
        created_at_utc=datetime.now(timezone.utc).isoformat(),predictor_script_sha256=sha256(Path(__file__)),**sources))
    log(f"Sealed predictions {args.detector}/{args.dataset}/{seed}: {len(target_p)} target rows")


def run(args):
    cfg=json.loads(args.config.read_text(encoding="utf-8"))
    rows=read_rows(args.source_rows,True)
    target_rows=read_rows(args.target_rows,False)
    positions=np.asarray([i for i,r in enumerate(rows) if r["role"]!="train"],dtype=np.int64)
    if args.detector=="hashing_sgd":
        xs,xt=hashed_features(args.source_rows,args.target_rows,args.feature_cache)
        fits=fit_hashing(args,rows,xs,cfg)
        for seed,(coef,intercept,receipt,checkpoint) in fits.items():
            ps=expit(np.asarray(xs[positions]@coef.T).ravel()+float(intercept[0]))
            pt=expit(np.asarray(xt@coef.T).ravel()+float(intercept[0]))
            write_prediction(args,seed,positions,ps,pt,None,None,receipt,checkpoint,args.feature_cache)
    elif args.detector=="frozen_codebert":
        xs=base_embeddings(args,args.source_tokens,args.source_embedding)
        arrays,receipt,checkpoint=fit_frozen(args,rows,xs)
        xt=base_embeddings(args,args.target_tokens,args.target_embedding)
        def probability(x):
            scaled=(x-arrays["mean"])/arrays["scale"]
            return expit((scaled@arrays["coef"].T).ravel()+float(arrays["intercept"][0]))
        write_prediction(args,13,positions,probability(xs[positions]),probability(xt),xs[positions],xt,receipt,checkpoint)
    else:
        import torch
        source=TokenCache(args.source_tokens,source=True)
        target=TokenCache(args.target_tokens,source=False)
        for seed in cfg["fine_tuned_codebert"]["seeds"]:
            receipt=args.training/f"seed-{seed}"/"training_manifest.json"
            checkpoint=receipt.parent/"best_model.pt"
            sealed=json.loads(receipt.read_text(encoding="utf-8"))
            if sealed["config_sha256"]!=sha256(args.config) or sealed["checkpoint_sha256"]!=sha256(checkpoint):
                raise ValueError("Training/checkpoint seal mismatch")
            set_seed(seed)
            model=CodeBertDetector(args.model_path).cuda()
            model.load_state_dict(torch.load(checkpoint,map_location="cuda",weights_only=True))
            ps,xs=predict(model,source,positions,embeddings=True)
            pt,xt=predict(model,target,np.arange(len(target.rows)),embeddings=True)
            write_prediction(args,seed,positions,ps,pt,xs,xt,receipt,checkpoint)
            del model
            torch.cuda.empty_cache()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--detector",choices=("hashing_sgd","frozen_codebert","finetuned_codebert"),required=True)
    parser.add_argument("--dataset",choices=("diversevul","megavul"),required=True)
    parser.add_argument("--source-rows",type=Path,required=True)
    parser.add_argument("--target-rows",type=Path,required=True)
    parser.add_argument("--source-tokens",type=Path)
    parser.add_argument("--target-tokens",type=Path)
    parser.add_argument("--model-path",type=Path)
    parser.add_argument("--feature-cache",type=Path)
    parser.add_argument("--source-embedding",type=Path)
    parser.add_argument("--target-embedding",type=Path)
    parser.add_argument("--training",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--config",type=Path,default=Path("configs/neural_validation_v3.json"))
    parser.add_argument("--dataset-config",type=Path)
    args=parser.parse_args()
    if args.detector=="hashing_sgd" and args.feature_cache is None:
        parser.error("hashing requires --feature-cache")
    if args.detector!="hashing_sgd" and any(v is None for v in (args.source_tokens,args.target_tokens,args.model_path)):
        parser.error("neural inference requires token caches and the pinned model path")
    run(args)

if __name__=="__main__":
    main()
