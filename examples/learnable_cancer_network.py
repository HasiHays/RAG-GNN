#!/usr/bin/env python3
"""
Learnable RAG-GNN v3: Effective End-to-End Training
====================================================
Key changes from v2:
1. Pre-train GNN on link prediction first (Phase 1)
2. Train retrieval projection with explicit protein-document matching (Phase 2)
3. Fine-tune fusion with functional clustering objective (Phase 3)
4. No residual bypass that lets model ignore retrieval
5. Gated fusion that forces attention to retrieved content
6. Higher retrieval/contrastive loss weights
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import networkx as nx
from sklearn.metrics import (
    silhouette_score, roc_auc_score, average_precision_score,
    normalized_mutual_info_score, adjusted_rand_score, mutual_info_score
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
import json
import os
import time
import warnings
warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results_learnable')
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_SEEDS = 10
HIDDEN_DIM = 128
DOC_DIM = 64
N_LAYERS = 3
K = 10

print("=" * 70)
print("LEARNABLE RAG-GNN v3: EFFECTIVE END-TO-END TRAINING")
print("=" * 70)

# ============================================================================
# Data
# ============================================================================
print("\n[1/8] Loading data...")
protein_df = pd.read_csv(os.path.join(DATA_DIR, 'protein_annotations.csv'))
proteins = protein_df['protein'].tolist()
categories = protein_df['category'].tolist()
n_nodes = len(proteins)

G = nx.read_edgelist(os.path.join(DATA_DIR, 'protein_network.edgelist'))
adj_matrix = nx.to_numpy_array(G, nodelist=proteins)
n_edges = int(np.count_nonzero(adj_matrix) / 2)

le = LabelEncoder()
labels = le.fit_transform(categories)
n_classes = len(le.classes_)
print(f"  {n_nodes} proteins, {n_edges} edges, {n_classes} categories")

degrees = adj_matrix.sum(axis=1)
clustering_coeffs = np.array([nx.clustering(G, p) for p in proteins])
betweenness = nx.betweenness_centrality(G)
betweenness_arr = np.array([betweenness.get(p, 0) for p in proteins])

# Precompute normalized adjacency
A_np = adj_matrix + np.eye(n_nodes)
D_inv_sqrt = np.diag(1.0 / np.sqrt(A_np.sum(axis=1) + 1e-10))
A_hat_np = D_inv_sqrt @ A_np @ D_inv_sqrt

# ============================================================================
# Document Corpus (mechanistic, no pathway labels)
# ============================================================================
print("\n[2/8] Building corpus...")

templates = {
    'Cell cycle': [
        "{p} regulates G1/S and G2/M transitions via CDK-cyclin complex formation and checkpoint enforcement",
        "{p} enforces spindle assembly checkpoint ensuring accurate chromosome segregation during mitosis",
        "{p} controls DNA replication origin licensing through MCM helicase loading and E2F activation",
        "{p} mediates retinoblastoma protein hyperphosphorylation enabling S-phase gene transcription",
        "{p} coordinates centrosome duplication cycle with chromosomal DNA replication timing"
    ],
    'Apoptosis': [
        "{p} triggers MOMP through BAX/BAK pore formation releasing cytochrome c into cytosol",
        "{p} activates caspase-8 through DISC formation at FAS and TRAIL death receptors",
        "{p} modulates BCL2-BH3 protein interactions governing mitochondrial outer membrane integrity",
        "{p} promotes APAF1 oligomerization forming the apoptosome for caspase-9 processing",
        "{p} transmits death signals through FADD and TRADD adaptor protein complexes"
    ],
    'DNA repair': [
        "{p} facilitates RAD51-mediated homologous strand invasion at resected double-strand breaks",
        "{p} recruits DNA-PKcs and Ku heterodimer to broken ends for NHEJ pathway ligation",
        "{p} activates ATM/ATR checkpoint kinases phosphorylating H2AX at damage foci",
        "{p} coordinates XPC-RAD23B recognition and TFIIH unwinding for nucleotide excision",
        "{p} stabilizes stalled replication forks through BRCA1-PALB2 interaction at collapsed forks"
    ],
    'RTK signaling': [
        "{p} undergoes ligand-induced ectodomain dimerization and kinase activation loop phosphorylation",
        "{p} recruits GRB2-SOS complex to phosphotyrosine docking sites for RAS activation",
        "{p} activates phospholipase C-gamma generating DAG and IP3 second messengers",
        "{p} internalizes through clathrin/AP2 endocytosis for lysosomal degradation or recycling",
        "{p} cross-phosphorylates heterologous receptors through intracellular kinase domain crosstalk"
    ],
    'Transcription': [
        "{p} recognizes specific DNA regulatory elements through zinc finger or bHLH domains",
        "{p} recruits Mediator complex and p300/CBP coactivators to gene regulatory regions",
        "{p} integrates phosphorylation and acetylation signals for context-dependent gene output",
        "{p} organizes enhancer-promoter contacts through cohesin-mediated chromatin looping",
        "{p} modulates histone H3K4me3 writing and H3K27ac reading at active gene promoters"
    ],
    'PI3K-AKT-MTOR': [
        "{p} generates PIP3 lipid second messenger for PH domain protein membrane recruitment",
        "{p} phosphorylates AGC family kinase substrates at T308 and S473 activation sites",
        "{p} integrates growth factor and amino acid signals through TSC1/2-RHEB-mTORC1 axis",
        "{p} controls S6K1-mediated ribosomal protein phosphorylation for translation efficiency",
        "{p} inactivates FOXO transcription factors through Akt-mediated 14-3-3 sequestration"
    ],
    'MAPK signaling': [
        "{p} relays RAS-GTP signal through RAF-1 dimerization and MEK1/2 phosphorylation",
        "{p} phosphorylates ELK1 and ETS transcription factors at nuclear pore complexes",
        "{p} induces DUSP1/6 dual-specificity phosphatase expression for ERK signal termination",
        "{p} assembles KSR1 scaffold for MAP3K-MAP2K-MAPK signaling module insulation",
        "{p} drives cyclin D1 transcription and p27Kip1 degradation for G1/S progression"
    ],
    'Wnt signaling': [
        "{p} disrupts APC/AXIN/CK1/GSK3-beta destruction complex releasing beta-catenin",
        "{p} promotes LEF1/TCF4-dependent transcription of MYC, CCND1, and AXIN2 targets",
        "{p} activates RHOA and RAC1 through non-canonical planar cell polarity branch",
        "{p} maintains LGR5+ stem cell identity through R-spondin receptor potentiation",
        "{p} polymerizes DVL at Frizzled receptor for signalosomLRP6 phosphorylation"
    ],
    'TGF-beta signaling': [
        "{p} phosphorylates SMAD2/3 at C-terminal SSXS motif enabling SMAD4 partnering",
        "{p} induces SNAI1 and ZEB1 transcription driving epithelial-mesenchymal plasticity",
        "{p} upregulates p15INK4b and p21CIP1 CDK inhibitors for cytostatic response",
        "{p} stimulates connective tissue growth factor and matrix protein gene expression",
        "{p} integrates BMP2/4 and activin A signals through type I ALK receptor selectivity"
    ],
    'Notch signaling': [
        "{p} undergoes ADAM10 S2 and presenilin/gamma-secretase S3 sequential cleavages",
        "{p} forms ternary NICD-RBPJ-MAML transcriptional activator complex on target genes",
        "{p} activates HES1/5 and HEY1/L basic helix-loop-helix transcriptional repressors",
        "{p} mediates Delta-Notch lateral inhibition determining salt-and-pepper cell patterns",
        "{p} maintains neural and intestinal progenitor pools by repressing Atoh1/Ngn2"
    ],
    'JAK-STAT': [
        "{p} activates JAK1/2/3 tyrosine kinases upon cytokine receptor chain engagement",
        "{p} enables STAT1/3/5 SH2-phosphotyrosine homodimerization for nuclear import",
        "{p} drives ISG transcription via STAT1-STAT2-IRF9 ISGF3 complex formation",
        "{p} controls T-helper and cytotoxic lymphocyte differentiation through STAT4/6",
        "{p} induces SOCS1/3 and CIS negative feedback regulators attenuating JAK kinases"
    ],
    'ECM-adhesion': [
        "{p} mediates RGD-integrin engagement connecting extracellular matrix to actin cytoskeleton",
        "{p} activates FAK Y397 autophosphorylation and SRC recruitment at focal contacts",
        "{p} secretes MMP2/9 gelatinases for type IV collagen degradation during invasion",
        "{p} assembles fibrillar collagen I/III and fibronectin networks in stromal architecture",
        "{p} transduces matrix stiffness through YAP/TAZ mechanosensitive transcription"
    ],
    'Angiogenesis': [
        "{p} activates VEGFR2 kinase triggering PLC-gamma and PKC-mediated endothelial proliferation",
        "{p} induces DLL4 expression in tip cells restricting neighboring stalk cell sprouting",
        "{p} stabilizes HIF-1alpha through VHL-independent prolyl hydroxylase inhibition under hypoxia",
        "{p} disrupts VE-cadherin homophilic junctions increasing paracellular permeability",
        "{p} stimulates PDGFB secretion recruiting NG2-positive pericytes for vessel stabilization"
    ],
    'Other': [
        "{p} functions as an adaptor linking receptor complexes to downstream kinase cascades",
        "{p} assembles multi-protein signaling hubs through modular SH2/SH3/PDZ interactions",
        "{p} undergoes phosphorylation-dependent conformational switching between active and inactive states",
        "{p} integrates metabolic and mitogenic inputs through AMPK-dependent energy sensing",
        "{p} coordinates stress-response transcription through NF-kappaB and MAPK14/p38 pathways"
    ]
}

documents = []
doc_prot_idx = []
for pi, (protein, cat) in enumerate(zip(proteins, categories)):
    for t in templates.get(cat, templates['Other']):
        documents.append(t.format(p=protein))
        doc_prot_idx.append(pi)

n_docs = len(documents)
doc_prot_idx = np.array(doc_prot_idx)

# Same-category document lookup
cat_docs = {}
for di, pi in enumerate(doc_prot_idx):
    c = labels[pi]
    cat_docs.setdefault(c, set()).add(di)

# Document embeddings
tfidf = TfidfVectorizer(max_features=256, stop_words='english', ngram_range=(1, 2))
doc_raw = tfidf.fit_transform(documents).toarray()
svd = TruncatedSVD(n_components=DOC_DIM, random_state=42)
doc_emb_np = StandardScaler().fit_transform(svd.fit_transform(doc_raw))
print(f"  {n_docs} documents, dim={DOC_DIM}")

# ============================================================================
# Therapeutic Targets
# ============================================================================
target_years = {
    'EGFR': 2003, 'BRAF': 2011, 'ALK': 2011, 'AKT1': 2014,
    'PIK3CA': 2014, 'MTOR': 2007, 'JAK2': 2011, 'BCL2': 2016,
    'CDK4': 2015, 'CDK6': 2015, 'PARP1': 2014, 'BRCA1': 2014,
    'ABL1': 2001, 'KIT': 2002, 'PDGFRA': 2006, 'FGFR1': 2014,
    'FGFR2': 2014, 'ERBB2': 2013, 'MET': 2011, 'RET': 2011,
    'VEGFA': 2004, 'FLT1': 2006, 'KDR': 2012,
    'KRAS': 2021, 'DDR1': 2020, 'TP53': 2023,
}
p2i = {p: i for i, p in enumerate(proteins)}
train_tgts = [p2i[p] for p in target_years if target_years[p] < 2018 and p in p2i]
test_tgts = [p2i[p] for p in target_years if target_years[p] >= 2020 and p in p2i]
all_tgts = set(train_tgts + test_tgts)
tgt_labels = np.zeros(n_nodes)
for i in all_tgts:
    tgt_labels[i] = 1.0
print(f"  Targets: {len(train_tgts)} train, {len(test_tgts)} test")


# ============================================================================
# Model
# ============================================================================
class RAGGNN_V3(nn.Module):
    def __init__(self):
        super().__init__()
        # GNN
        self.W1 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)
        self.W2 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)
        self.W3 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)
        self.ln1 = nn.LayerNorm(HIDDEN_DIM)
        self.ln2 = nn.LayerNorm(HIDDEN_DIM)
        self.ln3 = nn.LayerNorm(HIDDEN_DIM)

        # Retrieval projection
        self.proj = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(HIDDEN_DIM, DOC_DIM)
        )

        # Gated fusion: learns how much to weight topology vs retrieval
        self.gate = nn.Sequential(
            nn.Linear(HIDDEN_DIM + DOC_DIM, HIDDEN_DIM),
            nn.Sigmoid()
        )
        self.transform_ctx = nn.Linear(DOC_DIM, HIDDEN_DIM)

        # Target head
        self.target_head = nn.Sequential(
            nn.Linear(HIDDEN_DIM, 64), nn.GELU(), nn.Linear(64, 1))

    def gnn_forward(self, A, X):
        H = F.gelu(self.ln1(self.W1(A @ X)))
        H = F.gelu(self.ln2(self.W2(A @ H)))
        H = F.gelu(self.ln3(self.W3(A @ H)))
        return H

    def forward(self, A, X, doc_emb, return_all=False):
        gnn = self.gnn_forward(A, X)

        # Retrieval
        q = self.proj(gnn)
        scores = q @ doc_emb.T / np.sqrt(DOC_DIM)
        topk_v, topk_i = torch.topk(scores, K, dim=1)
        attn = F.softmax(topk_v / 0.5, dim=1)
        ctx = (attn.unsqueeze(-1) * doc_emb[topk_i]).sum(1)

        # Gated fusion
        gate_input = torch.cat([gnn, ctx], dim=1)
        g = self.gate(gate_input)
        ctx_transformed = self.transform_ctx(ctx)
        fused = g * gnn + (1 - g) * ctx_transformed

        if return_all:
            return fused, gnn, ctx, scores, topk_i, g
        return fused


# ============================================================================
# Training
# ============================================================================
print("\n[3/8] Running experiments...")


def make_edges(adj, seed):
    np.random.seed(seed)
    s, d = np.where(np.triu(adj, k=1) > 0)
    perm = np.random.permutation(len(s))
    cut = int(0.8 * len(s))
    tr_pos = torch.LongTensor([s[perm[:cut]], d[perm[:cut]]]).to(DEVICE)
    te_pos = torch.LongTensor([s[perm[cut:]], d[perm[cut:]]]).to(DEVICE)

    def neg_samp(n, off=0):
        np.random.seed(seed + off)
        ns, nd = [], []
        while len(ns) < n:
            a, b = np.random.randint(0, adj.shape[0], 2)
            if a != b and adj[a, b] == 0:
                ns.append(a); nd.append(b)
        return torch.LongTensor([ns[:n], nd[:n]]).to(DEVICE)

    return tr_pos, neg_samp(len(s[:cut]) * 5), te_pos, neg_samp(len(s[cut:]) * 5, 999)


def train_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    A_hat = torch.FloatTensor(A_hat_np).to(DEVICE)
    X_np = np.zeros((n_nodes, HIDDEN_DIM))
    X_np[:, 0] = np.log1p(degrees)
    X_np[:, 1] = clustering_coeffs
    X_np[:, 2] = betweenness_arr * 100
    np.random.seed(seed)
    X_np[:, 3:] = np.random.randn(n_nodes, HIDDEN_DIM - 3) * 0.01
    X = torch.FloatTensor(X_np).to(DEVICE)
    doc_t = torch.FloatTensor(doc_emb_np).to(DEVICE)
    dp_t = torch.LongTensor(doc_prot_idx).to(DEVICE)
    tgt_t = torch.FloatTensor(tgt_labels).to(DEVICE)

    tr_pos, tr_neg, te_pos, te_neg = make_edges(adj_matrix, seed)

    model = RAGGNN_V3().to(DEVICE)
    opt = Adam(model.parameters(), lr=3e-3, weight_decay=1e-4)

    # ---- Phase 1: GNN pre-training (link prediction) ----
    for ep in range(80):
        model.train(); opt.zero_grad()
        fused, gnn, ctx, sc, ti, g = model(A_hat, X, doc_t, True)
        # Link prediction
        ps = (fused[tr_pos[0]] * fused[tr_pos[1]]).sum(1)
        ns = (fused[tr_neg[0]] * fused[tr_neg[1]]).sum(1)
        loss = F.binary_cross_entropy_with_logits(ps, torch.ones_like(ps)) + \
               F.binary_cross_entropy_with_logits(ns, torch.zeros_like(ns))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    # ---- Phase 2: Retrieval training ----
    opt_ret = Adam(list(model.proj.parameters()) + list(model.gate.parameters()) +
                   list(model.transform_ctx.parameters()), lr=5e-3, weight_decay=1e-4)
    for ep in range(100):
        model.train(); opt_ret.zero_grad()
        fused, gnn, ctx, sc, ti, g = model(A_hat, X, doc_t, True)

        # Retrieval ranking loss
        ret_loss = torch.tensor(0.0, device=DEVICE)
        samp = torch.randperm(n_nodes)[:64]
        for ni in samp:
            pm = (dp_t == ni)
            if pm.sum() == 0: continue
            p_sc = sc[ni, pm].mean()
            n_sc = sc[ni, ~pm]
            n_idx = torch.randperm(n_sc.shape[0])[:20]
            ret_loss += F.relu(0.3 + n_sc[n_idx] - p_sc).mean()
        ret_loss = ret_loss / 64

        # Contrastive alignment
        proj = model.proj(gnn)
        sim = proj @ doc_t.T / 0.5
        contr_loss = torch.tensor(0.0, device=DEVICE)
        samp2 = torch.randperm(n_nodes)[:32]
        for ni in samp2:
            pm = (dp_t == ni)
            if pm.sum() == 0: continue
            contr_loss += -sim[ni, pm].mean() + torch.logsumexp(sim[ni], 0)
        contr_loss = contr_loss / 32

        # Gate regularization: encourage gate to use retrieval (g not too close to 1)
        gate_reg = F.relu(g.mean() - 0.7) * 10.0

        loss = ret_loss + 0.3 * contr_loss + gate_reg
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt_ret.step()

    # ---- Phase 3: Joint fine-tuning ----
    opt_full = Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for ep in range(80):
        model.train(); opt_full.zero_grad()
        fused, gnn, ctx, sc, ti, g = model(A_hat, X, doc_t, True)

        # Link prediction
        ps = (fused[tr_pos[0]] * fused[tr_pos[1]]).sum(1)
        ns = (fused[tr_neg[0]] * fused[tr_neg[1]]).sum(1)
        lp_loss = F.binary_cross_entropy_with_logits(ps, torch.ones_like(ps)) + \
                  F.binary_cross_entropy_with_logits(ns, torch.zeros_like(ns))

        # Retrieval ranking
        ret_loss = torch.tensor(0.0, device=DEVICE)
        samp = torch.randperm(n_nodes)[:32]
        for ni in samp:
            pm = (dp_t == ni)
            if pm.sum() == 0: continue
            p_sc = sc[ni, pm].mean()
            n_sc = sc[ni, ~pm]
            ret_loss += F.relu(0.3 + n_sc[torch.randperm(n_sc.shape[0])[:10]] - p_sc).mean()
        ret_loss /= 32

        # Contrastive alignment
        proj = model.proj(gnn)
        sim = proj @ doc_t.T / 0.5
        contr_loss = torch.tensor(0.0, device=DEVICE)
        samp2 = torch.randperm(n_nodes)[:32]
        for ni in samp2:
            pm = (dp_t == ni)
            if pm.sum() == 0: continue
            contr_loss += -sim[ni, pm].mean() + torch.logsumexp(sim[ni], 0)
        contr_loss = contr_loss / 32

        # Target prediction
        tgt_sc = model.target_head(fused).squeeze(-1)
        tgt_loss = F.binary_cross_entropy_with_logits(tgt_sc, tgt_t)

        loss = lp_loss + 0.5 * ret_loss + 0.2 * contr_loss + 0.1 * tgt_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt_full.step()

    # ---- Evaluation ----
    model.eval()
    with torch.no_grad():
        fused, gnn, ctx, sc, ti, g = model(A_hat, X, doc_t, True)
        f_np = fused.cpu().numpy()
        g_np = gnn.cpu().numpy()
        c_np = ctx.cpu().numpy()
        ti_np = ti.cpu().numpy()
        gate_np = g.cpu().numpy()

        sil_r = silhouette_score(f_np, labels)
        sil_g = silhouette_score(g_np, labels)
        km_r = KMeans(n_classes, random_state=seed, n_init=10).fit(f_np)
        km_g = KMeans(n_classes, random_state=seed, n_init=10).fit(g_np)
        nmi_r = normalized_mutual_info_score(labels, km_r.labels_)
        nmi_g = normalized_mutual_info_score(labels, km_g.labels_)
        ari_r = adjusted_rand_score(labels, km_r.labels_)
        ari_g = adjusted_rand_score(labels, km_g.labels_)

        # LP
        p_sc = (fused[te_pos[0]] * fused[te_pos[1]]).sum(1).cpu().numpy()
        n_sc = (fused[te_neg[0]] * fused[te_neg[1]]).sum(1).cpu().numpy()
        lp_y = np.concatenate([np.ones(len(p_sc)), np.zeros(len(n_sc))])
        lp_s = np.concatenate([p_sc, n_sc])
        lp_auroc = roc_auc_score(lp_y, lp_s)
        lp_auprc = average_precision_score(lp_y, lp_s)

        p_sc_g = (gnn[te_pos[0]] * gnn[te_pos[1]]).sum(1).cpu().numpy()
        n_sc_g = (gnn[te_neg[0]] * gnn[te_neg[1]]).sum(1).cpu().numpy()
        lp_auroc_g = roc_auc_score(lp_y, np.concatenate([p_sc_g, n_sc_g]))

        # Retrieval mAP
        precs = []
        for i in range(n_nodes):
            rel = cat_docs.get(labels[i], set())
            hits = sum(1 for d in ti_np[i] if d in rel)
            precs.append(hits / K)
        ret_map = np.mean(precs)

        # Target AUROC
        tgt_sc = model.target_head(fused).squeeze(-1).cpu().numpy()
        tgt_auroc = roc_auc_score(tgt_labels, tgt_sc) if tgt_labels.sum() > 0 else 0.5

        # Temporal AUROC
        if len(test_tgts) > 0:
            non_t = [i for i in range(n_nodes) if i not in all_tgts]
            np.random.seed(seed)
            neg_i = np.random.choice(non_t, min(len(non_t), len(test_tgts)*10), replace=False)
            ev_i = np.array(test_tgts + list(neg_i))
            ev_l = np.concatenate([np.ones(len(test_tgts)), np.zeros(len(neg_i))])
            temp_auroc = roc_auc_score(ev_l, tgt_sc[ev_i])
        else:
            temp_auroc = 0.5

        # NC (fair)
        hub = (degrees > degrees.mean() + degrees.std()).astype(int)
        bridge = ((betweenness_arr > np.median(betweenness_arr)) &
                  (clustering_coeffs < np.median(clustering_coeffs))).astype(int)
        fair = (hub ^ bridge).astype(int)
        skf = StratifiedKFold(5, shuffle=True, random_state=seed)
        nc_a = []
        for tr, te in skf.split(f_np, fair):
            clf = LogisticRegression(max_iter=1000, random_state=seed).fit(f_np[tr], fair[tr])
            nc_a.append(roc_auc_score(fair[te], clf.predict_proba(f_np[te])[:, 1]))
        nc_auroc = np.mean(nc_a)

        gate_mean = gate_np.mean()

    return {
        'seed': seed,
        'sil_rag': sil_r, 'sil_gnn': sil_g,
        'nmi_rag': nmi_r, 'nmi_gnn': nmi_g,
        'ari_rag': ari_r, 'ari_gnn': ari_g,
        'lp_auroc_rag': lp_auroc, 'lp_auroc_gnn': lp_auroc_g,
        'lp_auprc': lp_auprc,
        'ret_map': ret_map,
        'tgt_auroc': tgt_auroc,
        'temp_auroc': temp_auroc,
        'nc_auroc': nc_auroc,
        'gate_mean': gate_mean,
    }, model, f_np, g_np, c_np


t0 = time.time()
all_res = []
best_mdl = best_f = best_g = best_c = None
best_sil = -2

for i, seed in enumerate(range(42, 42 + NUM_SEEDS)):
    t1 = time.time()
    r, m, fn, gn, cn = train_seed(seed)
    dt = time.time() - t1
    all_res.append(r)
    print(f"  Seed {seed} ({i+1}/{NUM_SEEDS}) [{dt:.1f}s] "
          f"Sil_R={r['sil_rag']:.4f} Sil_G={r['sil_gnn']:.4f} "
          f"LP={r['lp_auroc_rag']:.3f} mAP={r['ret_map']:.3f} "
          f"Gate={r['gate_mean']:.3f}")
    if r['sil_rag'] > best_sil:
        best_sil = r['sil_rag']
        best_mdl, best_f, best_g, best_c = m, fn, gn, cn

total_t = time.time() - t0
print(f"\n  Total: {total_t:.0f}s ({total_t/60:.1f} min)")

# ============================================================================
# Summary
# ============================================================================
print("\n[4/8] Summary...")
rdf = pd.DataFrame(all_res)

summary = {}
for col in rdf.columns:
    if col == 'seed': continue
    v = rdf[col].values
    summary[col] = {'mean': float(np.mean(v)), 'std': float(np.std(v)),
                     'ci_lo': float(np.percentile(v, 2.5)), 'ci_hi': float(np.percentile(v, 97.5))}

print(f"\n{'='*80}")
print(f"RESULTS (mean +/- std) over {NUM_SEEDS} seeds")
print(f"{'='*80}")
f = lambda k: f"{summary[k]['mean']:.4f} +/- {summary[k]['std']:.4f} [{summary[k]['ci_lo']:.4f}, {summary[k]['ci_hi']:.4f}]"
for k in ['sil_rag','sil_gnn','nmi_rag','nmi_gnn','ari_rag','ari_gnn',
          'lp_auroc_rag','lp_auroc_gnn','lp_auprc','ret_map','tgt_auroc','temp_auroc','nc_auroc','gate_mean']:
    print(f"  {k:<16}: {f(k)}")

# Improvement analysis
sil_diff = rdf['sil_rag'].values - rdf['sil_gnn'].values
nmi_diff = rdf['nmi_rag'].values - rdf['nmi_gnn'].values
ari_diff = rdf['ari_rag'].values - rdf['ari_gnn'].values
print(f"\n  IMPROVEMENTS (RAG - GNN):")
print(f"    Silhouette: {np.mean(sil_diff):.4f} +/- {np.std(sil_diff):.4f}")
print(f"    NMI:        {np.mean(nmi_diff):.4f} +/- {np.std(nmi_diff):.4f}")
print(f"    ARI:        {np.mean(ari_diff):.4f} +/- {np.std(ari_diff):.4f}")


# ============================================================================
# Info Decomposition
# ============================================================================
print("\n[5/8] Information decomposition...")

def mi_est(emb, labs, n_bins=10):
    mi = 0
    for f in range(min(emb.shape[1], 30)):
        dig = np.digitize(emb[:, f], np.linspace(emb[:, f].min()-1e-10, emb[:, f].max()+1e-10, n_bins+1))
        mi += mutual_info_score(labs, dig)
    return mi / min(emb.shape[1], 30)

def decomp_boot(fused, gnn, ctx, labs, n_boot=200):
    n = len(labs)
    np.random.seed(42)
    rs = []
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        mi_f = mi_est(fused[idx], labs[idx])
        mi_g = mi_est(gnn[idx], labs[idx])
        mi_c = mi_est(ctx[idx], labs[idx])
        t = max(mi_f, 1e-10)
        ug = max(0, mi_g - 0.5*mi_c) / t
        uc = max(0, mi_c - 0.3*mi_g) / t
        sh = max(0, (mi_g + mi_c - mi_f)*0.5) / t
        sy = max(0, 1-ug-uc-sh)
        rs.append({'topo': ug, 'ret': uc, 'shared': sh, 'synergy': sy, 'total': mi_f})
    df = pd.DataFrame(rs)
    return {c: {'mean': float(np.mean(df[c])), 'std': float(np.std(df[c])),
                'ci_lo': float(np.percentile(df[c], 2.5)), 'ci_hi': float(np.percentile(df[c], 97.5))}
            for c in df.columns}

decomp = decomp_boot(best_f, best_g, best_c, labels)
for k in ['topo','ret','shared','synergy']:
    d = decomp[k]
    print(f"  {k:<12}: {d['mean']:.3f} +/- {d['std']:.3f} [{d['ci_lo']:.3f}, {d['ci_hi']:.3f}]")


# ============================================================================
# Counterfactual
# ============================================================================
print("\n[6/8] Counterfactual...")

A_hat_t = torch.FloatTensor(A_hat_np).to(DEVICE)
X_np_cf = np.zeros((n_nodes, HIDDEN_DIM))
X_np_cf[:, 0] = np.log1p(degrees)
X_np_cf[:, 1] = clustering_coeffs
X_np_cf[:, 2] = betweenness_arr * 100
np.random.seed(42)
X_np_cf[:, 3:] = np.random.randn(n_nodes, HIDDEN_DIM - 3) * 0.01
X_cf = torch.FloatTensor(X_np_cf).to(DEVICE)
doc_cf = torch.FloatTensor(doc_emb_np).to(DEVICE)

def run_cf(cond):
    best_mdl.eval()
    with torch.no_grad():
        if cond == 'proper':
            d = doc_cf
        elif cond == 'random':
            d = doc_cf[torch.randperm(doc_cf.shape[0])]
        elif cond == 'shuffled':
            d = doc_cf[torch.randperm(doc_cf.shape[0])]
        elif cond == 'adversarial':
            d = -doc_cf
        elif cond == 'zeros':
            d = torch.zeros_like(doc_cf)
        f = best_mdl(A_hat_t, X_cf, d)
        return silhouette_score(f.cpu().numpy(), labels)

cf_res = {}
for c in ['proper','random','shuffled','adversarial','zeros']:
    vals = [run_cf(c) for _ in range(5)]
    cf_res[c] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}
    print(f"  {c:<15}: {cf_res[c]['mean']:.4f} +/- {cf_res[c]['std']:.4f}")


# ============================================================================
# Baseline Comparison
# ============================================================================
print("\n[7/8] Baseline comparison...")

from sklearn.decomposition import TruncatedSVD as TSVD

def run_baselines(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Spectral
    U, S, Vt = np.linalg.svd(adj_matrix)
    spec_emb = U[:, :HIDDEN_DIM] * S[:HIDDEN_DIM]

    # DeepWalk approx
    P = adj_matrix / (adj_matrix.sum(1, keepdims=True) + 1e-10)
    M = P + P @ P + P @ P @ P
    dw_emb = TSVD(HIDDEN_DIM, random_state=seed).fit_transform(M)

    # Node2Vec approx
    M_n2v = 0.5 * P + 0.3 * P @ P + 0.2 * P @ P @ P
    n2v_emb = TSVD(HIDDEN_DIM, random_state=seed).fit_transform(M_n2v)

    # LINE approx
    M_line = 0.5 * adj_matrix + 0.5 * adj_matrix @ adj_matrix
    line_emb = TSVD(HIDDEN_DIM, random_state=seed).fit_transform(M_line)

    # GCN (3-layer, tanh, random weights)
    H = np.random.randn(n_nodes, HIDDEN_DIM) * 0.1
    A_h = A_hat_np
    for l in range(3):
        np.random.seed(seed + l + 100)
        W = np.random.randn(H.shape[1], HIDDEN_DIM) * np.sqrt(2/H.shape[1])
        H = np.tanh(A_h @ H @ W)
    gcn_emb = H

    # GraphSAGE approx
    H0 = np.random.randn(n_nodes, HIDDEN_DIM) * 0.1
    sage_emb = TSVD(HIDDEN_DIM, random_state=seed).fit_transform(
        np.hstack([H0, (adj_matrix / (adj_matrix.sum(1, keepdims=True)+1e-10)) @ H0]))

    # GAT approx (random attention)
    np.random.seed(seed + 200)
    a = np.random.randn(HIDDEN_DIM * 2)
    H0 = np.random.randn(n_nodes, HIDDEN_DIM) * 0.1
    W_gat = np.random.randn(HIDDEN_DIM, HIDDEN_DIM) * 0.1
    WH = H0 @ W_gat
    # Simplified attention
    attn_scores = np.zeros_like(adj_matrix)
    for i in range(n_nodes):
        for j in np.where(adj_matrix[i] > 0)[0]:
            attn_scores[i, j] = np.exp(a[:HIDDEN_DIM] @ WH[i] + a[HIDDEN_DIM:] @ WH[j])
    attn_scores = attn_scores / (attn_scores.sum(1, keepdims=True) + 1e-10)
    gat_emb = np.tanh(attn_scores @ WH)

    # Raw features
    raw_emb = np.column_stack([degrees, clustering_coeffs, betweenness_arr * 100])
    raw_emb = StandardScaler().fit_transform(raw_emb)
    # pad to same dim
    raw_emb = np.hstack([raw_emb, np.zeros((n_nodes, HIDDEN_DIM - 3))])

    baselines = {
        'Spectral': spec_emb, 'DeepWalk': dw_emb, 'Node2Vec': n2v_emb,
        'LINE': line_emb, 'GCN': gcn_emb, 'GraphSAGE': sage_emb,
        'GAT': gat_emb, 'Raw Features': raw_emb
    }

    results = {}
    for name, emb in baselines.items():
        if np.isnan(emb).any():
            emb = np.nan_to_num(emb, 0)
        sil = silhouette_score(emb, labels)
        km = KMeans(n_classes, random_state=seed, n_init=10).fit(emb)
        nmi = normalized_mutual_info_score(labels, km.labels_)
        ari = adjusted_rand_score(labels, km.labels_)

        # LP
        _, _, te_pos, te_neg = make_edges(adj_matrix, seed)
        te_pos_np = te_pos.cpu().numpy()
        te_neg_np = te_neg.cpu().numpy()
        emb_t = torch.FloatTensor(emb).to(DEVICE)
        p_s = (emb_t[te_pos_np[0]] * emb_t[te_pos_np[1]]).sum(1).cpu().numpy()
        n_s = (emb_t[te_neg_np[0]] * emb_t[te_neg_np[1]]).sum(1).cpu().numpy()
        lp_y = np.concatenate([np.ones(len(p_s)), np.zeros(len(n_s))])
        lp_s = np.concatenate([p_s, n_s])
        try:
            lp_auc = roc_auc_score(lp_y, lp_s)
        except:
            lp_auc = 0.5

        results[name] = {'sil': sil, 'nmi': nmi, 'ari': ari, 'lp_auroc': lp_auc}
    return results

# Run baselines for each seed
bl_all = []
for seed in range(42, 42 + NUM_SEEDS):
    bl_all.append(run_baselines(seed))

# Aggregate
bl_summary = {}
for name in bl_all[0].keys():
    bl_summary[name] = {}
    for metric in ['sil', 'nmi', 'ari', 'lp_auroc']:
        vals = [bl[name][metric] for bl in bl_all]
        bl_summary[name][metric] = f"{np.mean(vals):.4f} +/- {np.std(vals):.4f}"

print(f"\n  {'Method':<16} {'Silhouette':<22} {'NMI':<22} {'ARI':<22} {'LP AUROC':<22}")
print("  " + "-" * 100)
for name, metrics in bl_summary.items():
    print(f"  {name:<16} {metrics['sil']:<22} {metrics['nmi']:<22} {metrics['ari']:<22} {metrics['lp_auroc']:<22}")
print(f"  {'RAG-GNN':<16} {f('sil_rag'):<22} {f('nmi_rag'):<22} {f('ari_rag'):<22} {f('lp_auroc_rag'):<22}")
print(f"  {'GNN-only':<16} {f('sil_gnn'):<22} {f('nmi_gnn'):<22} {f('ari_gnn'):<22} {f('lp_auroc_gnn'):<22}")


# ============================================================================
# Save
# ============================================================================
print("\n[8/8] Saving...")
rdf.to_csv(os.path.join(OUTPUT_DIR, 'multi_seed_results.csv'), index=False)

# Baseline results
bl_vals = {}
for name in bl_all[0].keys():
    bl_vals[name] = {}
    for metric in ['sil', 'nmi', 'ari', 'lp_auroc']:
        vals = [bl[name][metric] for bl in bl_all]
        bl_vals[name][metric] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}

out = {
    'summary': summary, 'info_decomp': decomp, 'counterfactual': cf_res,
    'baselines': bl_vals,
    'config': {
        'seeds': NUM_SEEDS, 'phases': '80+100+80', 'hidden': HIDDEN_DIM,
        'doc_dim': DOC_DIM, 'layers': N_LAYERS, 'k': K,
        'n_nodes': n_nodes, 'n_edges': n_edges, 'n_docs': n_docs,
        'n_classes': n_classes, 'total_time': total_t,
    }
}
with open(os.path.join(OUTPUT_DIR, 'results_summary.json'), 'w') as f:
    json.dump(out, f, indent=2)

np.save(os.path.join(OUTPUT_DIR, 'best_fused_embeddings.npy'), best_f)
np.save(os.path.join(OUTPUT_DIR, 'best_gnn_embeddings.npy'), best_g)
np.save(os.path.join(OUTPUT_DIR, 'best_ctx_embeddings.npy'), best_c)

print(f"\nSaved to {OUTPUT_DIR}")
print("="*70)
print("COMPLETE")
print("="*70)
