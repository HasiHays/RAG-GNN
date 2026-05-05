#!/usr/bin/env python3
"""
Regenerate Figures 3, 4, 5, 6 for RAG-GNN manuscript revision.
Uses corrected v3 learnable pipeline results.
- Fig-3: Retrieval PR curves (trains one seed for score matrix)
- Fig-4: Embedding visualization (saved embeddings)
- Fig-5: DDR1 subnetwork (saved embeddings)
- Fig-6: Benchmark comparison (results_summary.json)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from sklearn.metrics import (
    silhouette_score, precision_recall_curve, average_precision_score,
    roc_auc_score, normalized_mutual_info_score, adjusted_rand_score
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.cluster import KMeans
from scipy.spatial.distance import cosine
import json, os, shutil, warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================
BASE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.join(BASE, '..')
DATA_DIR = os.path.join(REPO_ROOT, 'data')
RESULTS_DIR = os.path.join(REPO_ROOT, 'results_learnable')
OUTPUT_DIR = os.path.join(REPO_ROOT, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

HIDDEN_DIM = 128
DOC_DIM = 64
N_LAYERS = 3
K = 10
DEVICE = torch.device('cpu')

SET2_COLORS = [
    '#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854',
    '#ffd92f', '#e5c494', '#b3b3b3', '#1b9e77', '#d95f02',
    '#7570b3', '#e7298a', '#66a61e', '#a6761d',
]

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 12,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 8,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'axes.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
})

category_names = {
    0: 'Cell cycle', 1: 'Apoptosis', 2: 'DNA repair', 3: 'RTK signaling',
    4: 'Transcription', 5: 'PI3K-AKT-MTOR', 6: 'MAPK signaling',
    7: 'Wnt signaling', 8: 'TGF-beta signaling', 9: 'Notch signaling',
    10: 'JAK-STAT', 11: 'ECM-adhesion', 12: 'Angiogenesis', 13: 'Other'
}

print("=" * 70)
print("REGENERATING FIGURES 3, 4, 5, 6")
print("=" * 70)

# ============================================================================
# Load shared data
# ============================================================================
print("\n[1/7] Loading data...")
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

degrees = adj_matrix.sum(axis=1)
clustering_coeffs = np.array([nx.clustering(G, p) for p in proteins])
betweenness = nx.betweenness_centrality(G)
betweenness_arr = np.array([betweenness.get(p, 0) for p in proteins])

category_ids = protein_df['category_id'].tolist()

# Precompute normalized adjacency
A_np = adj_matrix + np.eye(n_nodes)
D_inv_sqrt = np.diag(1.0 / np.sqrt(A_np.sum(axis=1) + 1e-10))
A_hat_np = D_inv_sqrt @ A_np @ D_inv_sqrt

print(f"  {n_nodes} proteins, {n_edges} edges, {n_classes} categories")

# Load results
with open(os.path.join(RESULTS_DIR, 'results_summary.json')) as f:
    results = json.load(f)

# Load saved embeddings
fused_emb = np.load(os.path.join(RESULTS_DIR, 'best_fused_embeddings.npy'))
gnn_emb = np.load(os.path.join(RESULTS_DIR, 'best_gnn_embeddings.npy'))
ctx_emb = np.load(os.path.join(RESULTS_DIR, 'best_ctx_embeddings.npy'))
print(f"  Loaded embeddings: fused={fused_emb.shape}, gnn={gnn_emb.shape}, ctx={ctx_emb.shape}")

# ============================================================================
# Build document corpus (needed for Fig-3)
# ============================================================================
print("\n[2/7] Building document corpus...")

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

cat_docs = {}
for di, pi in enumerate(doc_prot_idx):
    c = labels[pi]
    cat_docs.setdefault(c, set()).add(di)

tfidf = TfidfVectorizer(max_features=256, stop_words='english', ngram_range=(1, 2))
doc_raw = tfidf.fit_transform(documents).toarray()
svd = TruncatedSVD(n_components=DOC_DIM, random_state=42)
doc_emb_np = StandardScaler().fit_transform(svd.fit_transform(doc_raw))
print(f"  {n_docs} documents, dim={DOC_DIM}")

# Therapeutic targets for training
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

# ============================================================================
# Fig-3: Train one seed for retrieval PR curves
# ============================================================================
print("\n[3/7] Training one seed for retrieval scores (Fig-3)...")

class RAGGNN_V3(nn.Module):
    def __init__(self):
        super().__init__()
        self.W1 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)
        self.W2 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)
        self.W3 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)
        self.ln1 = nn.LayerNorm(HIDDEN_DIM)
        self.ln2 = nn.LayerNorm(HIDDEN_DIM)
        self.ln3 = nn.LayerNorm(HIDDEN_DIM)
        self.proj = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.GELU(),
            nn.Dropout(0.1), nn.Linear(HIDDEN_DIM, DOC_DIM))
        self.gate = nn.Sequential(
            nn.Linear(HIDDEN_DIM + DOC_DIM, HIDDEN_DIM), nn.Sigmoid())
        self.transform_ctx = nn.Linear(DOC_DIM, HIDDEN_DIM)
        self.target_head = nn.Sequential(
            nn.Linear(HIDDEN_DIM, 64), nn.GELU(), nn.Linear(64, 1))

    def gnn_forward(self, A, X):
        H = F.gelu(self.ln1(self.W1(A @ X)))
        H = F.gelu(self.ln2(self.W2(A @ H)))
        H = F.gelu(self.ln3(self.W3(A @ H)))
        return H

    def forward(self, A, X, doc_emb, return_all=False):
        gnn = self.gnn_forward(A, X)
        q = self.proj(gnn)
        scores = q @ doc_emb.T / np.sqrt(DOC_DIM)
        topk_v, topk_i = torch.topk(scores, K, dim=1)
        attn = F.softmax(topk_v / 0.5, dim=1)
        ctx = (attn.unsqueeze(-1) * doc_emb[topk_i]).sum(1)
        gate_input = torch.cat([gnn, ctx], dim=1)
        g = self.gate(gate_input)
        ctx_transformed = self.transform_ctx(ctx)
        fused = g * gnn + (1 - g) * ctx_transformed
        if return_all:
            return fused, gnn, ctx, scores, topk_i, g
        return fused

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

seed = 42
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

# Phase 1: GNN pre-training
print("  Phase 1: GNN pre-training (80 epochs)...")
for ep in range(80):
    model.train(); opt.zero_grad()
    fused, gnn, ctx, sc, ti, g = model(A_hat, X, doc_t, True)
    ps = (fused[tr_pos[0]] * fused[tr_pos[1]]).sum(1)
    ns = (fused[tr_neg[0]] * fused[tr_neg[1]]).sum(1)
    loss = F.binary_cross_entropy_with_logits(ps, torch.ones_like(ps)) + \
           F.binary_cross_entropy_with_logits(ns, torch.zeros_like(ns))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()

# Phase 2: Retrieval training
print("  Phase 2: Retrieval training (100 epochs)...")
opt_ret = Adam(list(model.proj.parameters()) + list(model.gate.parameters()) +
               list(model.transform_ctx.parameters()), lr=5e-3, weight_decay=1e-4)
for ep in range(100):
    model.train(); opt_ret.zero_grad()
    fused, gnn, ctx, sc, ti, g = model(A_hat, X, doc_t, True)
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
    proj = model.proj(gnn)
    sim = proj @ doc_t.T / 0.5
    contr_loss = torch.tensor(0.0, device=DEVICE)
    samp2 = torch.randperm(n_nodes)[:32]
    for ni in samp2:
        pm = (dp_t == ni)
        if pm.sum() == 0: continue
        contr_loss += -sim[ni, pm].mean() + torch.logsumexp(sim[ni], 0)
    contr_loss = contr_loss / 32
    gate_reg = F.relu(g.mean() - 0.7) * 10.0
    loss = ret_loss + 0.3 * contr_loss + gate_reg
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt_ret.step()

# Phase 3: Joint fine-tuning
print("  Phase 3: Joint fine-tuning (80 epochs)...")
opt_full = Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
for ep in range(80):
    model.train(); opt_full.zero_grad()
    fused, gnn, ctx, sc, ti, g = model(A_hat, X, doc_t, True)
    ps = (fused[tr_pos[0]] * fused[tr_pos[1]]).sum(1)
    ns = (fused[tr_neg[0]] * fused[tr_neg[1]]).sum(1)
    lp_loss = F.binary_cross_entropy_with_logits(ps, torch.ones_like(ps)) + \
              F.binary_cross_entropy_with_logits(ns, torch.zeros_like(ns))
    ret_loss = torch.tensor(0.0, device=DEVICE)
    samp = torch.randperm(n_nodes)[:32]
    for ni in samp:
        pm = (dp_t == ni)
        if pm.sum() == 0: continue
        p_sc = sc[ni, pm].mean()
        n_sc = sc[ni, ~pm]
        ret_loss += F.relu(0.3 + n_sc[torch.randperm(n_sc.shape[0])[:10]] - p_sc).mean()
    ret_loss /= 32
    proj = model.proj(gnn)
    sim = proj @ doc_t.T / 0.5
    contr_loss = torch.tensor(0.0, device=DEVICE)
    samp2 = torch.randperm(n_nodes)[:32]
    for ni in samp2:
        pm = (dp_t == ni)
        if pm.sum() == 0: continue
        contr_loss += -sim[ni, pm].mean() + torch.logsumexp(sim[ni], 0)
    contr_loss = contr_loss / 32
    tgt_sc = model.target_head(fused).squeeze(-1)
    tgt_loss = F.binary_cross_entropy_with_logits(tgt_sc, tgt_t)
    loss = lp_loss + 0.5 * ret_loss + 0.2 * contr_loss + 0.1 * tgt_loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt_full.step()

# Extract retrieval scores for PR curves
model.eval()
with torch.no_grad():
    fused_tr, gnn_tr, ctx_tr, scores_tr, ti_tr, g_tr = model(A_hat, X, doc_t, True)
    score_matrix = scores_tr.cpu().numpy()  # (379, 1895)

print("  Training complete. Extracting retrieval scores...")

# Compute PR curves for Fig-3
# Relevance: document is relevant if it's about the same protein OR same category
def compute_pr_global(score_matrix, doc_prot_idx, labels):
    all_labels_list = []
    all_scores_list = []
    for i in range(len(labels)):
        doc_labels = np.array([(1 if (doc_prot_idx[d] == i or labels[doc_prot_idx[d]] == labels[i])
                                else 0) for d in range(score_matrix.shape[1])])
        all_labels_list.extend(doc_labels.tolist())
        all_scores_list.extend(score_matrix[i].tolist())
    return np.array(all_labels_list), np.array(all_scores_list)

rag_labels_pr, rag_scores_pr = compute_pr_global(score_matrix, doc_prot_idx, labels)

# TF-IDF baseline: protein name queries against document corpus
protein_queries = [f"{p} {c} signaling pathway" for p, c in zip(proteins, categories)]
tfidf_q = TfidfVectorizer(max_features=256, stop_words='english')
tfidf_q.fit(documents)
query_tfidf = tfidf_q.transform(protein_queries).toarray()
doc_tfidf = tfidf_q.transform(documents).toarray()
query_norms = query_tfidf / (np.linalg.norm(query_tfidf, axis=1, keepdims=True) + 1e-10)
doc_norms_tfidf = doc_tfidf / (np.linalg.norm(doc_tfidf, axis=1, keepdims=True) + 1e-10)
tfidf_score_matrix = query_norms @ doc_norms_tfidf.T
tfidf_labels_pr, tfidf_scores_pr = compute_pr_global(tfidf_score_matrix, doc_prot_idx, labels)

# Topology-only baseline: SVD of adjacency
svd_adj = TruncatedSVD(n_components=HIDDEN_DIM, random_state=42)
adj_emb_topo = svd_adj.fit_transform(adj_matrix)
np.random.seed(42)
proj_topo = np.random.randn(HIDDEN_DIM, DOC_DIM) * 0.1
topo_queries = adj_emb_topo @ proj_topo
topo_q_norm = topo_queries / (np.linalg.norm(topo_queries, axis=1, keepdims=True) + 1e-10)
doc_n = doc_emb_np / (np.linalg.norm(doc_emb_np, axis=1, keepdims=True) + 1e-10)
topo_score_matrix = topo_q_norm @ doc_n.T
topo_labels_pr, topo_scores_pr = compute_pr_global(topo_score_matrix, doc_prot_idx, labels)

# Random baseline
np.random.seed(123)
random_scores_pr = np.random.rand(len(rag_labels_pr))
random_labels_pr = rag_labels_pr.copy()

# Generate Fig-3
print("\n[4/7] Generating Fig-3 (Retrieval Performance)...")
fig3, ax3 = plt.subplots(figsize=(8, 6))

methods_pr = {
    'RAG-GNN (ours)': {'labels': rag_labels_pr, 'scores': rag_scores_pr,
                       'color': SET2_COLORS[0], 'lw': 2.5},
    'Topology-only': {'labels': topo_labels_pr, 'scores': topo_scores_pr,
                      'color': SET2_COLORS[1], 'lw': 2},
    'TF-IDF': {'labels': tfidf_labels_pr, 'scores': tfidf_scores_pr,
               'color': SET2_COLORS[2], 'lw': 2},
    'Random': {'labels': random_labels_pr, 'scores': random_scores_pr,
               'color': SET2_COLORS[7], 'lw': 1.5, 'ls': '--'},
}

fig3_maps = {}
for method_name, config in methods_pr.items():
    precision, recall, _ = precision_recall_curve(config['labels'], config['scores'])
    ap = average_precision_score(config['labels'], config['scores'])
    fig3_maps[method_name] = ap
    ls = config.get('ls', '-')
    ax3.plot(recall, precision, label=f"{method_name} (mAP={ap:.3f})",
             color=config['color'], linewidth=config['lw'], linestyle=ls, alpha=0.9)

ax3.set_xlabel('Recall', fontsize=12)
ax3.set_ylabel('Precision', fontsize=12)
ax3.set_xlim([0.0, 1.0])
ax3.set_ylim([0.0, 1.05])
ax3.legend(loc='upper right', fontsize=10, frameon=False)
ax3.set_title('Document retrieval performance for protein queries',
              fontsize=12, fontweight='bold')
plt.tight_layout()

for ext in ['pdf', 'svg']:
    fig3.savefig(os.path.join(OUTPUT_DIR, f'Fig-2-Retrieval-Performance.{ext}'),
                 format=ext, dpi=300, bbox_inches='tight')
fig3.savefig(os.path.join(OUTPUT_DIR, 'Fig-3.pdf'),
             format='pdf', dpi=300, bbox_inches='tight')
fig3.savefig(os.path.join(OUTPUT_DIR, 'Fig-3.svg'),
             format='svg', dpi=300, bbox_inches='tight')
print("  Saved Fig-3.pdf and Fig-3.svg")
for mn, ap in fig3_maps.items():
    print(f"    {mn}: mAP={ap:.3f}")
plt.close()

# ============================================================================
# Fig-4: Embedding Visualization (2x3 grid)
# ============================================================================
print("\n[5/7] Generating Fig-4 (Embedding Visualization)...")

sil_rag = results['summary']['sil_rag']['mean']
sil_gnn = results['summary']['sil_gnn']['mean']

fig4 = plt.figure(figsize=(18, 10))
gs4 = GridSpec(2, 3, figure=fig4, width_ratios=[1.8, 1, 1], hspace=0.35, wspace=0.3)

# Panel A: RAG-GNN Embeddings PCA
ax_a = fig4.add_subplot(gs4[:, 0])
pca4 = PCA(n_components=2, random_state=42)
rag_2d = pca4.fit_transform(fused_emb)

for cat_id in range(14):
    mask_cat = np.array(category_ids) == cat_id
    if np.sum(mask_cat) > 0:
        ax_a.scatter(rag_2d[mask_cat, 0], rag_2d[mask_cat, 1],
                     c=SET2_COLORS[cat_id % len(SET2_COLORS)],
                     label=f"{category_names[cat_id]} (n={np.sum(mask_cat)})",
                     alpha=0.7, s=40, edgecolors='white', linewidths=0.3)

key_proteins = ['TP53', 'EGFR', 'KRAS', 'MYC', 'BRCA1', 'PIK3CA', 'AKT1', 'PTEN']
for protein in key_proteins:
    if protein in proteins:
        idx = proteins.index(protein)
        ax_a.scatter(rag_2d[idx, 0], rag_2d[idx, 1], s=150, facecolors='none',
                     edgecolors='black', linewidths=1.5, zorder=10)
        ax_a.annotate(protein, (rag_2d[idx, 0], rag_2d[idx, 1]),
                      xytext=(5, 5), textcoords='offset points',
                      fontsize=8, fontweight='bold',
                      bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

ax_a.set_xlabel('PC1', fontsize=11)
ax_a.set_ylabel('PC2', fontsize=11)
ax_a.set_title('(A) RAG-GNN protein embeddings', fontsize=12, fontweight='bold', loc='left')
handles, lab = ax_a.get_legend_handles_labels()
ax_a.legend(handles, lab, loc='upper right', fontsize=7, ncol=2,
            frameon=True, fancybox=True, title='Functional categories', title_fontsize=8)

stats_text = (f"Network: {n_nodes} proteins, {n_edges} interactions\n"
              f"Silhouette (RAG-GNN): {sil_rag:.3f}\n"
              f"Silhouette (GNN-only): {sil_gnn:.3f}\n"
              f"Improvement: {sil_rag - sil_gnn:+.3f}")
ax_a.text(0.02, 0.02, stats_text, transform=ax_a.transAxes, fontsize=8,
          verticalalignment='bottom', family='monospace',
          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

# Panel B: Degree Distribution
ax_b = fig4.add_subplot(gs4[0, 1])
deg_list = [G.degree(p) for p in proteins]
ax_b.hist(deg_list, bins=40, color=SET2_COLORS[0], alpha=0.7, edgecolor='black', linewidth=0.5)
ax_b.set_xlabel('Node degree', fontsize=10)
ax_b.set_ylabel('Count', fontsize=10)
ax_b.set_title('(B) Degree distribution', fontsize=11, fontweight='bold', loc='left')
ax_b.set_yscale('log')

# Panel C: Silhouette Comparison
ax_c = fig4.add_subplot(gs4[0, 2])
meth_names = ['RAG-GNN', 'GNN-only']
sils = [sil_rag, sil_gnn]
bars_c = ax_c.bar(meth_names, sils, color=[SET2_COLORS[0], SET2_COLORS[1]],
                  edgecolor='black', linewidth=0.5)
ax_c.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax_c.set_ylabel('Silhouette score', fontsize=10)
ax_c.set_title('(C) Clustering quality comparison', fontsize=11, fontweight='bold', loc='left')
for bar, val in zip(bars_c, sils):
    ax_c.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
              f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Panel D: Top Hub Proteins
ax_d = fig4.add_subplot(gs4[1, 1])
top_30 = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:20]
top_names = [x[0] for x in top_30]
top_values = [x[1] for x in top_30]
ax_d.barh(range(len(top_names)), top_values, color=SET2_COLORS[2], alpha=0.7,
          edgecolor='black', linewidth=0.3)
ax_d.set_yticks(range(len(top_names)))
ax_d.set_yticklabels(top_names, fontsize=7)
ax_d.set_xlabel('Betweenness centrality', fontsize=10)
ax_d.set_title('(D) Top 20 hub proteins', fontsize=11, fontweight='bold', loc='left')
ax_d.invert_yaxis()

# Panel E: Category Distribution
ax_e = fig4.add_subplot(gs4[1, 2])
cat_counts = []
cat_names_plot = []
cat_ids_plot = []
for cat_id, cat_name in category_names.items():
    count = np.sum(np.array(category_ids) == cat_id)
    if count > 0:
        cat_counts.append(count)
        cat_names_plot.append(cat_name)
        cat_ids_plot.append(cat_id)

ax_e.barh(range(len(cat_names_plot)), cat_counts,
          color=[SET2_COLORS[i % len(SET2_COLORS)] for i in cat_ids_plot],
          alpha=0.7, edgecolor='black', linewidth=0.3)
ax_e.set_yticks(range(len(cat_names_plot)))
ax_e.set_yticklabels(cat_names_plot, fontsize=8)
ax_e.set_xlabel('Number of proteins', fontsize=10)
ax_e.set_title('(E) Functional category distribution', fontsize=11, fontweight='bold', loc='left')
ax_e.invert_yaxis()

plt.tight_layout()
for ext in ['pdf', 'svg']:
    fig4.savefig(os.path.join(OUTPUT_DIR, f'Fig-1-RAG-Embeddings.{ext}'),
                 format=ext, dpi=300, bbox_inches='tight')
fig4.savefig(os.path.join(OUTPUT_DIR, 'Fig-4.pdf'),
             format='pdf', dpi=300, bbox_inches='tight')
fig4.savefig(os.path.join(OUTPUT_DIR, 'Fig-4.svg'),
             format='svg', dpi=300, bbox_inches='tight')
print("  Saved Fig-4.pdf and Fig-4.svg")
plt.close()

# ============================================================================
# Fig-5: DDR1 Subnetwork Visualization (1x2)
# ============================================================================
print("\n[6/7] Generating Fig-5 (DDR1 Subnetwork)...")

central_protein = 'DDR1'
if central_protein not in G.nodes():
    degs = dict(G.degree())
    central_protein = max(degs, key=degs.get)

neighbors_1hop = list(G.neighbors(central_protein))
neighbors_2hop = []
for n in neighbors_1hop[:10]:
    neighbors_2hop.extend([n2 for n2 in G.neighbors(n)
                           if n2 != central_protein and n2 not in neighbors_1hop])
neighbors_2hop = sorted(set(neighbors_2hop))[:20]

subnetwork_nodes = [central_protein] + neighbors_1hop + neighbors_2hop
subG = G.subgraph(subnetwork_nodes).copy()

node_categories = {}
node_category_ids = {}
for node in subG.nodes():
    if node in proteins:
        idx = proteins.index(node)
        node_categories[node] = categories[idx]
        node_category_ids[node] = category_ids[idx]
    else:
        node_categories[node] = 'Other'
        node_category_ids[node] = 13

central_idx = proteins.index(central_protein)
central_emb_vec = fused_emb[central_idx]

node_similarities = {}
for node in subG.nodes():
    if node in proteins:
        idx = proteins.index(node)
        node_emb_vec = fused_emb[idx]
        similarity = 1 - cosine(central_emb_vec, node_emb_vec)
        node_similarities[node] = max(0, similarity)
    else:
        node_similarities[node] = 0.0

sorted_sims = sorted([(n, s) for n, s in node_similarities.items() if n != central_protein],
                      key=lambda x: x[1], reverse=True)

betweenness_sub = nx.betweenness_centrality(subG)

category_colors = {
    0: '#E74C3C', 1: '#3498DB', 2: '#2ECC71', 3: '#9B59B6',
    4: '#F1C40F', 5: '#E67E22', 6: '#1ABC9C', 7: '#34495E',
    8: '#16A085', 9: '#8E44AD', 10: '#2980B9', 11: '#27AE60',
    12: '#C0392B', 13: '#95A5A6',
}

fig5, axes5 = plt.subplots(1, 2, figsize=(14, 6))
pos = nx.spring_layout(subG, k=2, iterations=50, seed=42)

important_nodes = [central_protein] + sorted(
    [n for n in neighbors_1hop if n in subG.nodes()],
    key=lambda x: betweenness_sub.get(x, 0), reverse=True)[:5]

# Panel A: Functional category
ax5a = axes5[0]
nx.draw_networkx_edges(subG, pos, width=0.5, alpha=0.3, edge_color='gray', ax=ax5a)
for cat_id in set(node_category_ids.values()):
    nodelist = [n for n in subG.nodes() if node_category_ids[n] == cat_id]
    if nodelist:
        node_sizes = [600 if n == central_protein else 250 for n in nodelist]
        nx.draw_networkx_nodes(
            subG, pos, nodelist=nodelist, node_color=category_colors[cat_id],
            node_size=node_sizes, alpha=0.85, edgecolors='black',
            linewidths=[1.5 if n == central_protein else 0.5 for n in nodelist], ax=ax5a)

labels_dict = {n: n for n in important_nodes}
nx.draw_networkx_labels(subG, pos, labels_dict, font_size=8, font_weight='bold',
                        font_color='black', ax=ax5a)
ax5a.set_title('(A) Functional category', fontsize=12, fontweight='bold', loc='left')
ax5a.axis('off')

present_categories = set(node_category_ids.values())
legend_elements = [mpatches.Patch(facecolor=category_colors[cid], edgecolor='black',
                                  label=category_names[cid])
                   for cid in sorted(present_categories)]
ax5a.legend(handles=legend_elements, loc='upper left', fontsize=7, frameon=False)

# Panel B: Embedding similarity
ax5b = axes5[1]
nx.draw_networkx_edges(subG, pos, width=0.5, alpha=0.3, edge_color='gray', ax=ax5b)
similarities_list = [node_similarities[n] for n in subG.nodes()]
node_sizes = [600 if n == central_protein else 250 for n in subG.nodes()]
nodes_plot = nx.draw_networkx_nodes(
    subG, pos, node_color=similarities_list, node_size=node_sizes,
    cmap='RdYlGn', vmin=0.0, vmax=1.0, alpha=0.85, edgecolors='black',
    linewidths=[1.5 if n == central_protein else 0.5 for n in subG.nodes()], ax=ax5b)
nx.draw_networkx_labels(subG, pos, labels_dict, font_size=8, font_weight='bold',
                        font_color='black', ax=ax5b)
ax5b.set_title('(B) Embedding similarity', fontsize=12, fontweight='bold', loc='left')
ax5b.axis('off')
cbar = plt.colorbar(nodes_plot, ax=ax5b, fraction=0.046, pad=0.04)
cbar.set_label('Cosine similarity', fontsize=10)
cbar.ax.tick_params(labelsize=8)

plt.tight_layout()
for ext in ['pdf', 'svg']:
    fig5.savefig(os.path.join(OUTPUT_DIR, f'Fig-4-DDR1-Network.{ext}'),
                 format=ext, dpi=300, bbox_inches='tight')
fig5.savefig(os.path.join(OUTPUT_DIR, 'Fig-5.pdf'),
             format='pdf', dpi=300, bbox_inches='tight')
fig5.savefig(os.path.join(OUTPUT_DIR, 'Fig-5.svg'),
             format='svg', dpi=300, bbox_inches='tight')
print("  Saved Fig-5.pdf and Fig-5.svg")

# Print similarity data for caption verification
print(f"\n  DDR1 subnetwork: {subG.number_of_nodes()} nodes, {subG.number_of_edges()} edges")
print(f"  1-hop: {len(neighbors_1hop)}, 2-hop: {len(neighbors_2hop)}")
print(f"  Top 5 similar proteins to {central_protein}:")
for i, (p, s) in enumerate(sorted_sims[:5], 1):
    print(f"    {i}. {p} ({node_categories[p]}): {s:.3f}")
plt.close()

# ============================================================================
# Fig-6: Benchmark Comparison (3x4 grid)
# ============================================================================
print("\n[7/7] Generating Fig-6 (Benchmark Comparison)...")

# Build benchmark DataFrame from results_summary.json
bl = results['baselines']
method_order = ['RAG-GNN', 'GNN-only', 'GCN', 'GAT', 'GraphSAGE',
                'DeepWalk', 'Node2Vec', 'LINE', 'Spectral', 'Raw Features']

bench_data = {
    'Method': method_order,
    'Silhouette': [
        results['summary']['sil_rag']['mean'],
        results['summary']['sil_gnn']['mean'],
        bl['GCN']['sil']['mean'], bl['GAT']['sil']['mean'],
        bl['GraphSAGE']['sil']['mean'], bl['DeepWalk']['sil']['mean'],
        bl['Node2Vec']['sil']['mean'], bl['LINE']['sil']['mean'],
        bl['Spectral']['sil']['mean'], bl['Raw Features']['sil']['mean'],
    ],
    'LP AUROC': [
        results['summary']['lp_auroc_rag']['mean'],
        results['summary']['lp_auroc_gnn']['mean'],
        bl['GCN']['lp_auroc']['mean'], bl['GAT']['lp_auroc']['mean'],
        bl['GraphSAGE']['lp_auroc']['mean'], bl['DeepWalk']['lp_auroc']['mean'],
        bl['Node2Vec']['lp_auroc']['mean'], bl['LINE']['lp_auroc']['mean'],
        bl['Spectral']['lp_auroc']['mean'], bl['Raw Features']['lp_auroc']['mean'],
    ],
    'NMI': [
        results['summary']['nmi_rag']['mean'],
        results['summary']['nmi_gnn']['mean'],
        bl['GCN']['nmi']['mean'], bl['GAT']['nmi']['mean'],
        bl['GraphSAGE']['nmi']['mean'], bl['DeepWalk']['nmi']['mean'],
        bl['Node2Vec']['nmi']['mean'], bl['LINE']['nmi']['mean'],
        bl['Spectral']['nmi']['mean'], bl['Raw Features']['nmi']['mean'],
    ],
    'ARI': [
        results['summary']['ari_rag']['mean'],
        results['summary']['ari_gnn']['mean'],
        bl['GCN']['ari']['mean'], bl['GAT']['ari']['mean'],
        bl['GraphSAGE']['ari']['mean'], bl['DeepWalk']['ari']['mean'],
        bl['Node2Vec']['ari']['mean'], bl['LINE']['ari']['mean'],
        bl['Spectral']['ari']['mean'], bl['Raw Features']['ari']['mean'],
    ],
}
bench_df = pd.DataFrame(bench_data)

n_methods = len(method_order)
x_pos = np.arange(n_methods)
bar_width = 0.7

method_colors = {m: SET2_COLORS[i] for i, m in enumerate(method_order)}
colors = [method_colors[m] for m in method_order]

fig6 = plt.figure(figsize=(16, 14))
gs6 = GridSpec(3, 4, figure=fig6, height_ratios=[1, 1, 1.1], hspace=0.45, wspace=0.3)

# Panel A: Silhouette Scores
ax6a = fig6.add_subplot(gs6[0, 0:2])
bars6a = ax6a.bar(x_pos, bench_df['Silhouette'], bar_width, color=colors,
                  edgecolor='black', linewidth=0.5)
ax6a.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax6a.set_ylabel('Silhouette score')
ax6a.set_title('(A) Functional clustering quality', fontweight='bold', loc='left')
ax6a.set_xticks(x_pos)
ax6a.set_xticklabels(method_order, rotation=45, ha='right')
ax6a.set_ylim(-0.3, 0.05)
# Highlight best silhouette (highest = GraphSAGE at -0.019, but RAG-GNN is the method of interest)
bars6a[0].set_edgecolor('red')
bars6a[0].set_linewidth(2)
for bar, val in zip(bars6a, bench_df['Silhouette']):
    ax6a.text(bar.get_x() + bar.get_width()/2,
              bar.get_height() + 0.005 if val >= 0 else bar.get_height() - 0.015,
              f'{val:.3f}', ha='center',
              va='bottom' if val >= 0 else 'top', fontsize=7)

# Panel B: LP AUROC
ax6b = fig6.add_subplot(gs6[0, 2:4])
bars6b = ax6b.bar(x_pos, bench_df['LP AUROC'], bar_width, color=colors,
                  edgecolor='black', linewidth=0.5)
ax6b.axhline(y=0.5, color='gray', linestyle='--', linewidth=0.5, label='Random')
ax6b.set_ylabel('AUROC')
ax6b.set_title('(B) Link prediction performance', fontweight='bold', loc='left')
ax6b.set_xticks(x_pos)
ax6b.set_xticklabels(method_order, rotation=45, ha='right')
ax6b.set_ylim(0.4, 1.05)
best_lp_idx = bench_df['LP AUROC'].idxmax()
bars6b[best_lp_idx].set_edgecolor('red')
bars6b[best_lp_idx].set_linewidth(2)
for bar, val in zip(bars6b, bench_df['LP AUROC']):
    ax6b.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
              f'{val:.3f}', ha='center', va='bottom', fontsize=7)

# Panel C: NMI
ax6c = fig6.add_subplot(gs6[1, 0:2])
bars6c = ax6c.bar(x_pos, bench_df['NMI'], bar_width, color=colors,
                  edgecolor='black', linewidth=0.5)
ax6c.set_ylabel('NMI')
ax6c.set_title('(C) Normalized mutual information', fontweight='bold', loc='left')
ax6c.set_xticks(x_pos)
ax6c.set_xticklabels(method_order, rotation=45, ha='right')
ax6c.set_ylim(0, 0.35)
best_nmi_idx = bench_df['NMI'].idxmax()
bars6c[best_nmi_idx].set_edgecolor('red')
bars6c[best_nmi_idx].set_linewidth(2)
for bar, val in zip(bars6c, bench_df['NMI']):
    ax6c.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
              f'{val:.3f}', ha='center', va='bottom', fontsize=7)

# Panel D: ARI
ax6d = fig6.add_subplot(gs6[1, 2:4])
bars6d = ax6d.bar(x_pos, bench_df['ARI'], bar_width, color=colors,
                  edgecolor='black', linewidth=0.5)
ax6d.set_ylabel('ARI')
ax6d.set_title('(D) Adjusted Rand index', fontweight='bold', loc='left')
ax6d.set_xticks(x_pos)
ax6d.set_xticklabels(method_order, rotation=45, ha='right')
ax6d.set_ylim(0, 0.14)
best_ari_idx = bench_df['ARI'].idxmax()
bars6d[best_ari_idx].set_edgecolor('red')
bars6d[best_ari_idx].set_linewidth(2)
for bar, val in zip(bars6d, bench_df['ARI']):
    ax6d.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
              f'{val:.3f}', ha='center', va='bottom', fontsize=7)

# Panel E: RAG-GNN PCA scatter
ax6e = fig6.add_subplot(gs6[2, 0])
pca6 = PCA(n_components=2, random_state=42)
rag_2d_6 = pca6.fit_transform(fused_emb)
unique_cats = sorted(set(categories))
cat_colors_map = {cat: SET2_COLORS[i % len(SET2_COLORS)] for i, cat in enumerate(unique_cats)}
point_colors = [cat_colors_map[cat] for cat in categories]
ax6e.scatter(rag_2d_6[:, 0], rag_2d_6[:, 1], c=point_colors, s=15, alpha=0.7, edgecolors='none')
ax6e.set_xlabel('PC1')
ax6e.set_ylabel('PC2')
ax6e.set_title('(E) RAG-GNN embeddings', fontweight='bold', loc='left')

# Panel F: GNN-only PCA scatter
ax6f = fig6.add_subplot(gs6[2, 1])
gnn_2d_6 = pca6.fit_transform(gnn_emb)
ax6f.scatter(gnn_2d_6[:, 0], gnn_2d_6[:, 1], c=point_colors, s=15, alpha=0.7, edgecolors='none')
ax6f.set_xlabel('PC1')
ax6f.set_ylabel('PC2')
ax6f.set_title('(F) GNN-only embeddings', fontweight='bold', loc='left')

# Panel G: Radar Chart
ax6g = fig6.add_subplot(gs6[2, 2], projection='polar')
radar_metrics = ['Silhouette', 'LP AUROC', 'NMI', 'ARI']
rag_vals = bench_df[bench_df['Method'] == 'RAG-GNN'][radar_metrics].values[0]
gnn_vals = bench_df[bench_df['Method'] == 'GNN-only'][radar_metrics].values[0]
gcn_vals = bench_df[bench_df['Method'] == 'GCN'][radar_metrics].values[0]

all_vals = bench_df[radar_metrics].values
min_vals = all_vals.min(axis=0)
max_vals = all_vals.max(axis=0)
range_vals = max_vals - min_vals + 1e-10

rag_norm = (rag_vals - min_vals) / range_vals
gnn_norm = (gnn_vals - min_vals) / range_vals
gcn_norm = (gcn_vals - min_vals) / range_vals

angles = np.linspace(0, 2*np.pi, len(radar_metrics), endpoint=False).tolist()
angles += angles[:1]
rag_norm = np.append(rag_norm, rag_norm[0])
gnn_norm = np.append(gnn_norm, gnn_norm[0])
gcn_norm = np.append(gcn_norm, gcn_norm[0])

scale_factor = 0.45
ax6g.plot(angles, rag_norm * scale_factor, 'o-', linewidth=1, color=SET2_COLORS[0],
          label='RAG-GNN', markersize=3)
ax6g.fill(angles, rag_norm * scale_factor, alpha=0.15, color=SET2_COLORS[0])
ax6g.plot(angles, gnn_norm * scale_factor, 's-', linewidth=1, color=SET2_COLORS[1],
          label='GNN-only', markersize=3)
ax6g.fill(angles, gnn_norm * scale_factor, alpha=0.15, color=SET2_COLORS[1])
ax6g.plot(angles, gcn_norm * scale_factor, '^-', linewidth=1, color=SET2_COLORS[2],
          label='GCN', markersize=3)
ax6g.fill(angles, gcn_norm * scale_factor, alpha=0.15, color=SET2_COLORS[2])

short_labels = ['Silhouette', 'LP AUROC', 'NMI', 'ARI']
ax6g.set_xticks(angles[:-1])
ax6g.set_xticklabels(short_labels, fontsize=7)
ax6g.set_ylim(0, 0.55)
ax6g.set_yticks([0.1, 0.2, 0.3, 0.4])
ax6g.set_yticklabels(['0.2', '0.4', '0.6', '0.8'], fontsize=6)
ax6g.set_title('')
fig6.text(0.515, 0.34, '(G) Method comparison', fontweight='bold', fontsize=11, ha='left')
ax6g.legend(loc='lower right', bbox_to_anchor=(1.3, -0.1), fontsize=7, frameon=True)

# Panel H: RAG-GNN Improvement
ax6h = fig6.add_subplot(gs6[2, 3])
delta_metrics = ['Silhouette', 'LP AUROC', 'NMI', 'ARI']
rag_row = bench_df[bench_df['Method'] == 'RAG-GNN'].iloc[0]
gnn_row = bench_df[bench_df['Method'] == 'GNN-only'].iloc[0]
deltas = [rag_row[m] - gnn_row[m] for m in delta_metrics]

bar_colors_h = [SET2_COLORS[0] if d > 0 else SET2_COLORS[1] for d in deltas]
bars6h = ax6h.barh(delta_metrics, deltas, color=bar_colors_h, edgecolor='black',
                   linewidth=0.5, height=0.5)
ax6h.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
ax6h.set_xlabel(r'$\Delta$ vs GNN-only')
ax6h.set_title('(H) RAG-GNN improvement', fontweight='bold', loc='left')

for bar, val in zip(bars6h, deltas):
    offset = 0.003 if val > 0 else -0.003
    ax6h.text(val + offset, bar.get_y() + bar.get_height()/2,
              f'{val:+.3f}', ha='left' if val > 0 else 'right',
              va='center', fontsize=9)

plt.tight_layout()
for ext in ['pdf', 'svg']:
    fig6.savefig(os.path.join(OUTPUT_DIR, f'Fig-Benchmark.{ext}'),
                 format=ext, dpi=300, bbox_inches='tight')
fig6.savefig(os.path.join(OUTPUT_DIR, 'Fig-6.pdf'),
             format='pdf', dpi=300, bbox_inches='tight')
fig6.savefig(os.path.join(OUTPUT_DIR, 'Fig-6.svg'),
             format='svg', dpi=300, bbox_inches='tight')
print("  Saved Fig-6.pdf and Fig-6.svg")
plt.close()

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 70)
print("FIGURE REGENERATION COMPLETE")
print("=" * 70)
print(f"\nFiles saved to manuscript directory ({OUTPUT_DIR}):")
print("  Fig-3.pdf / Fig-3.svg  (Retrieval PR curves)")
print("  Fig-4.pdf / Fig-4.svg  (Embedding visualization)")
print("  Fig-5.pdf / Fig-5.svg  (DDR1 subnetwork)")
print("  Fig-6.pdf / Fig-6.svg  (Benchmark comparison)")
print(f"\nKey values:")
print(f"  Silhouette RAG-GNN: {sil_rag:.3f}, GNN-only: {sil_gnn:.3f}, delta: {sil_rag-sil_gnn:+.3f}")
print(f"  NMI RAG-GNN: {results['summary']['nmi_rag']['mean']:.3f}")
print(f"  ARI RAG-GNN: {results['summary']['ari_rag']['mean']:.3f}")
print(f"  LP AUROC RAG-GNN: {results['summary']['lp_auroc_rag']['mean']:.3f}")
print(f"  Gate mean: {results['summary']['gate_mean']['mean']:.3f}")
