import nltk 
import tiktoken
from collections import Counter
from nltk.corpus import stopwords
import string
from typing import Sequence, Any, Optional
import pandas as pd
import time
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.model_selection import cross_validate
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from core.graf import *

TIKTOKEN_ENCODING = tiktoken.get_encoding('cl100k_base')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def tokenize_tiktoken(text: str) -> list[str]:
    token_bytes = [
        TIKTOKEN_ENCODING.decode_single_token_bytes(token)
        for token in TIKTOKEN_ENCODING.encode(text)
    ]

    tokens = [token.decode('utf-8', errors='replace').strip().lower() for token in token_bytes]
    return [token for token in tokens if token]

def filter_tokens(tokens:Sequence[str],
                  *,
                  remove_stopwords: bool = True,
                  remove_punctuation: bool = True,
                  custom_stopwords: set[str] = None,
                  custom_punctuation: set[str] = None,
                  lowercase_for_counting: bool = True,
                  language: str = 'english',
                  ) -> list[str]:
    
    stopwords_set = set(stopwords.words(language)) if custom_stopwords is None else custom_stopwords if remove_stopwords is True else None
    punctuation_set = set(string.punctuation) if custom_punctuation is None else custom_punctuation if remove_punctuation is True else None

    result = []

    for token in tokens:
        token = str(token).strip()
        if token == '':
            continue
        if remove_punctuation and (token in punctuation_set or all(c in punctuation_set for c in token)):
            continue
        if remove_stopwords and (token in stopwords_set or all(c in stopwords_set for c in token.split())):
            continue

        result.append(token.lower()) if lowercase_for_counting else result.append(token)

    return result

def count_tokens(
        texts,
        filter:bool = True,
    ):
    total_counts = Counter()
    for tokens in texts:
        if filter:
            tokens = filter_tokens(tokens)
        total_counts.update(tokens)

    return pd.Series(total_counts).sort_values(ascending=False)

def generate_ngrams(
        tokens: list[str],
        n: int = 2,
        filter: bool = True
):    
    if n < 1:
        raise  ValueError("n must be > 1")
    
    if filter:
            tokens = filter_tokens(tokens)

    if len(tokens) < n:
        return []
    
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def training_model_cv(
        model: BaseEstimator,
        X: Any,
        y: Any,
        model_name: str = 'model',
        preprocessor: Optional[Any] = None,
        cv: int = 5,
   ):
    if isinstance(preprocessor, Pipeline):
        steps = preprocessor.steps.copy()
        steps.append(("model", model))
        pipline = Pipeline(steps)
    elif preprocessor is not None:
        pipline = Pipeline([
            ("preprocessor", preprocessor),
            ("models", model), 
        ])
    else:
        pipline = model

    scoring = {
            "accuracy": "accuracy",
            "precision": make_scorer(precision_score, average="macro", zero_division=0),
            "recall": make_scorer(recall_score, average="macro", zero_division=0),
            "f1": make_scorer(f1_score, average="macro", zero_division=0),
            "roc_auc": "roc_auc_ovr"
        }
    
    start_time = time.time()
    cv_results = cross_validate(
        pipline,
        X,
        y,
        cv=cv,
        scoring=scoring,
        return_train_score=False
    )
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Learning time of {model_name}: {training_time}")

    cv_results = {k: sum(m)/len(m) for k, m in cv_results.items()}

    return cv_results

def training_dict_of_models_cv(
    models: dict,
    X: Any,
    y: Any,
    preprocessor: Optional[Any] = None,
    cv: Optional[int] = 5,
    ):
    
    metrics_dict = {}

    for name, model in models.items():
        metrics_dict[name] = training_model_cv(model=model, X=X, y=y, model_name=name, preprocessor=preprocessor, cv=cv)
        if isinstance(preprocessor, Pipeline):
            steps = preprocessor.steps.copy()
            steps.append(("model", model))
            pipeline = Pipeline(steps=steps)
        else:
            pipeline = model
            
        plot_feature_importance(pipeline, X, y, model_name=name)

    sns.heatmap(pd.DataFrame(metrics_dict).iloc[2:], annot=True, cmap='RdBu_r')

def evaluate(model, dl, criterion, device):
    model.eval()
    all_preds, all_labels = [], []
    val_loss = 0
    with torch.no_grad():
        for batch in dl:
            ids  = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            lbl  = batch['labels'].to(device)
            with torch.amp.autocast('cuda'):
                logits = model(ids, mask)
                if isinstance(logits, dict):
                    logits = logits.logits
                val_loss += criterion(logits, lbl).item()
            preds = (torch.sigmoid(logits) > 0.5).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(lbl.cpu().numpy())
    all_preds  = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    f1_list = []
    for i in range(all_labels.shape[1]):
        f1_list.append(f1_score(all_labels[:, i], all_preds[:, i]))
    return val_loss / len(dl), f1_list

def training(
        model: Any,
        train_data_loader: DataLoader,
        val_data_loader: DataLoader | None,
        optimizer: Any,
        epoches: int,
        criterion: Any,
        tokenizator: Any,
        accum_steps = 4,
        warmup_frac = 0.1,
        device: str = 'cpu',
        save_path: str = 'news_classifier_model',
        early_stop: int = 5
        
):
    
    total_steps  = (len(train_data_loader) // accum_steps) * epoches
    warmup_steps = int(total_steps * warmup_frac)
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler       = torch.amp.GradScaler('cuda')
    
    best_f1 = 0
    epoch_without_improvement = 0

    for epoch in range(1, epoches + 1):
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        pbar = tqdm(enumerate(train_data_loader), total=len(train_data_loader), desc=f'Epoch {epoch}/{epoches}')
        for step, batch in pbar:
                ids      = batch['input_ids'].to(device)
                mask     = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                with torch.amp.autocast('cuda'):
                        logits = model(ids, mask)
                        if isinstance(logits, dict):
                               logits = logits['logits']
                        loss   = criterion(logits, labels) / accum_steps

                scaler.scale(loss).backward()

                if (step + 1) % accum_steps == 0:
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                        scheduler.step()
                        optimizer.zero_grad()

                total_loss += loss.item() * accum_steps
                pbar.set_postfix(loss=f'{loss.item() * accum_steps:.4f}')

        if isinstance(val_data_loader, DataLoader):
                val_loss, f1_list = evaluate(model, val_data_loader, criterion, device)
                mean_f1 = sum(f1_list) / len(f1_list)
                f1_scores = pd.DataFrame(f1_list, index=tokenizator.categories_[0], columns=['f1']).transpose()
                print(f'Epoch {epoch} | train_loss={total_loss/len(train_data_loader):.4f} '
                      f'| val_loss={val_loss:.4f} | val_f1_mean={mean_f1:.4f} |')
                display(f1_scores)
        else:
               print(f'Epoch {epoch} | train_loss={total_loss/len(train_data_loader):.4f} ')
        
        if mean_f1 > best_f1:
            best_f1 = mean_f1
            torch.save(model.state_dict(), save_path + '_best.pt')
            print(f'  → Сохранена лучшая модель (mean F1={mean_f1:.4f})')
        else:
            epoch_without_improvement += 1
        
        if epoch_without_improvement > early_stop:
            return f1_scores

    return f1_scores


if __name__ == "__main__":
    print("ok")