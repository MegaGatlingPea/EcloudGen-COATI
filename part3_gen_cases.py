import pickle
import random
import argparse
import numpy as np
from rdkit import Chem
import rdkit.Chem.QED
import matplotlib
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
import torch
from torch.utils.data import DataLoader
from coati.models.io.coati import load_e3gnn_smiles_clip_e2e
from coati.models.regression.basic_due import basic_due
from coati.data.dataset import ecloud_dataset
from coati.common.util import batch_indexable
from coati.math_tools.altair_plots import roc_plot
from coati.generative.coati_purifications import force_decode_valid_batch, purify_vector, embed_smiles
from coati.generative.embed_altair import embed_altair

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True

#---------------------------case 1---------------------------#
def basic_generation(args):
    DEVICE = torch.device(args.device)
    encoder, tokenizer = load_e3gnn_smiles_clip_e2e(
        freeze=True,
        device=DEVICE,
        # model parameters to load.
        doc_url=args.model,
    )

    dataset = ecloud_dataset(args.dataset)
    epoch_iter = DataLoader(dataset = dataset, batch_size = 1, shuffle=False)
    out=[]
    all=0
    for batch_data in epoch_iter:
        eclouds = torch.Tensor(batch_data["eclouds"]).to(torch.float).to(DEVICE)
        smiles_list = encoder.eclouds_to_2d_batch(eclouds, tokenizer, '[SMILES]', noise_scale=args.noise)
        for s in smiles_list:
            all += 1
            if Chem.MolFromSmiles(s)!=None:
                out.append(s)

    if out!=[]:
        with open(args.output,'w') as f:
            for s in out:
                f.write(s+'\n')

    print('valid: ', len(out), len(out)/all)

#---------------------------case 2---------------------------#
def generation_near_a_given_mol(args):
    DEVICE = torch.device(args.device)
    encoder, tokenizer = load_e3gnn_smiles_clip_e2e(
        freeze=True,
        device=DEVICE,
        # model parameters to load.
        doc_url=args.model,
    )

    dataset = ecloud_dataset(args.dataset)
    ref = dataset[4] # only choose one for test
    ecloud = torch.Tensor(ref["eclouds"]).to(torch.float).to(DEVICE).unsqueeze(0)
    raw_token = torch.Tensor(ref["raw_tokens"]).to(torch.float).to(DEVICE).unsqueeze(0)
    num_to_gen = 50
    smiles_near_ref = encoder.ecloud_and_token_to_2d_batch(
    eclouds = ecloud,
    tokens = raw_token.long(),
    tokenizer = tokenizer,
    num_to_gen = num_to_gen, 
    noise_scale = args.noise,
    )
    with open(args.output,'w') as f:
        for s in smiles_near_ref:
            if Chem.MolFromSmiles(s)!=None:
                f.write(s+'\n')

#---------------------------case 3---------------------------#
def embed_and_score_in_batches_regression(
    records,
    encoder,
    tokenizer,
    batch_size=128,
    score=True,
    smiles_field="smiles",
):
    # A helper function to compute embeddings from the model encoder and
    # rdkit properties of molecules in batches. The input list of dict records
    # is modified in place.

    print("Embedding and scoring iterable from smiles.")
    batch_iter = batch_indexable(records, batch_size)
    num_batches = len(records) // batch_size
    with torch.no_grad():
        for i, batch in enumerate(batch_iter):
            print(f"batch: {i}/{num_batches}")
            try:
                batch_mols = [Chem.MolFromSmiles(row[smiles_field]) for row in batch]
                batch_smiles = [Chem.MolToSmiles(m) for m in batch_mols]
                batch_tokens = torch.tensor(
                    [
                        tokenizer.tokenize_text("[SMILES]" + s + "[STOP]", pad=True)
                        if s != "*"
                        else tokenizer.tokenize_text("[SMILES]C[STOP]", pad=True)
                        for s in batch_smiles
                    ],
                    device=encoder.device,
                    dtype=torch.int,
                )
                batch_embeds = encoder.encode_tokens(batch_tokens, tokenizer)
                if score:
                    batch_logp = [rdkit.Chem.Crippen.MolLogP(m) for m in batch_mols]
                    batch_qed = [rdkit.Chem.QED.qed(m) for m in batch_mols]
                if len(batch) < 2:
                    batch[0]["emb_smiles"] = batch_embeds[0].detach().cpu().numpy()
                    if score:
                        batch[0]["qed"] = batch_qed[0]
                        batch[0]["logp"] = batch_logp[0]
                        batch[0]["smiles"] = batch_smiles[0]
                else:
                    for k, r in enumerate(batch):
                        batch[k]["emb_smiles"] = batch_embeds[k].detach().cpu().numpy()
                        if score:
                            batch[k]["qed"] = batch_qed[k]
                            batch[k]["logp"] = batch_logp[k]
                            batch[k]["smiles"] = batch_smiles[k]
            except Exception as e:
                print(e)
                continue

def get_due_plot(due_result, y_field="qed", save_name="regression_plot.png"):
    """Plots the DUE model regressed results."""

    # xs are the true values, ys are the predicted values, dys are the errors.
    xs, ys, dys = due_result

    fig, ax = plt.subplots(figsize=(11, 6))
    n_to_plot = 30000
    # plt.style.use("seaborn-v0_8-pape")
    ax.errorbar(
        (xs[:n_to_plot]),
        (ys[:n_to_plot]),
        yerr=dys[:n_to_plot],
        fmt="o",
        color="black",
        ecolor="lightgray",
        elinewidth=3,
        capsize=0,
    )
    plt.xlabel("True " + y_field)
    plt.ylabel("Regressed " + y_field)
    plt.savefig(save_name)

def train_regression(args, y_field="qed"):
    DEVICE = torch.device(args.device)
    encoder, tokenizer = load_e3gnn_smiles_clip_e2e(
        freeze=True,
        device=DEVICE,
        # model parameters to load.
        doc_url=args.model,
    )

    with open(args.smiles) as f:
        smiles = [line.strip('\n') for line in f][:10000]
    random.shuffle(smiles)
    subset = [{"smiles": s, "source": "demo_mols"} for s in smiles]
    embed_and_score_in_batches_regression(subset, encoder, tokenizer)
    qed_model, qed_res = basic_due(
        subset,
        x_field="emb_smiles",
        y_field=y_field,
        save_as="regression_"+y_field+".pkl",
        continue_training=True,
        steps=1e4,
        random_seed=args.seed
        )
    qed_model = qed_model.to(DEVICE)
    get_due_plot(qed_res, y_field=y_field, save_name=y_field+"_regression_plot.png")

#---------------------------case 4---------------------------#
def bump_potential(V, bumps=[], radius=0.125, height=80.0, vec_dim=256):
    """
    Explore space by using gaussian bump potentials when the vector isn't
    changing a lot.
    """

    if len(bumps) < 1:
        return torch.zeros(1, device=V.device)

    bump_potential = (
        height
        * (
            (
                torch.distributions.multivariate_normal.MultivariateNormal(
                    torch.stack(bumps, 0).to(V.device),
                    radius * torch.eye(vec_dim).to(V.device),
                ).log_prob(V)
                + (vec_dim / 2) * (np.log(2 * torch.pi) + np.log(radius))
            ).exp()
        ).sum()
    )

    return bump_potential

def coati_metadynamics(
    init_emb_vec,
    objective_fcn,
    encoder,
    tokenizer,
    constraint_functions=[],  # enforced to == 0 by lagrange multipliers.
    log_functions=[],
    bump_radius=0.125 * 4,
    bump_height=80.0 * 16,
    nsteps=4000,
    save_traj_history=None,
):
    """
    Minimize an objective function in coati space.
    Purifies the vector as it goes along.
    The purified vector satisfies:
      vec \approx purify_vector(vec)

    contraint_functions: list of dict
        routines returning 'constraint_name' : tensor pairs.
    log_functions: list of dict
        routines returning 'value_name' : tensor pairs.

    Returns:
        history: (list of dict). Trajectory history.
    """
    vec = torch.nn.Parameter(init_emb_vec.to(encoder.device))
    vec.requires_grad = True

    # setup optimizer (SGD, Adam, etc.)
    params = [vec]
    for _ in constraint_functions.keys():
        params.append(torch.nn.Parameter(100 * torch.ones_like(vec[:1])))

    # optimizer = torch.optim.Adam(params,lr = 2e-3)
    optimizer = torch.optim.SGD(params, lr=2e-3)

    smiles = force_decode_valid_batch(init_emb_vec, encoder, tokenizer)
    history = [
        {
            "emb": vec.flatten().detach().cpu().numpy(),
            "name": 0,
            "smiles": smiles,
            "library": "opt",
            "activity": objective_fcn(vec).item(),
            **{
                c: constraint_functions[c](vec).detach().cpu().numpy()
                for c in constraint_functions
            },
            **{c: log_functions[c](vec).detach().cpu().numpy() for c in log_functions},
        }
    ]

    # no bumps are initialized.
    bumps = []
    last_bump = 0
    save_every = 25
    project_every = 15
    for k in range(nsteps):
        if k % project_every == 0 and k > 0:
            vec.data = 0.4 * vec.data + 0.6 * purify_vector(
                vec.data, encoder, tokenizer, n_rep=50
            )

        optimizer.zero_grad()
        activity = objective_fcn(vec)

        constraint_values = []
        for f in constraint_functions.keys():
            constraint_values.append(constraint_functions[f](vec))
        if len(constraint_values):
            constraint_term = torch.sum(torch.concat(constraint_values))
        else:
            constraint_term = torch.zeros_like(activity)

        # add a bump_term to the loss (=0 if no bumps).
        bump_term = bump_potential(vec, bumps, radius=bump_radius, height=bump_height)

        loss = activity + constraint_term + bump_term
        loss.backward()  # retain_graph=True)

        if k % save_every == 0:
            # Try to decode the vector here into a molecule!.
            smiles = force_decode_valid_batch(vec, encoder, tokenizer)

            history.append(
                {
                    "emb": vec.flatten().detach().cpu().numpy(),
                    "name": k,
                    "smiles": smiles,
                    "library": "opt",
                    "loss": loss.detach().cpu().item(),
                    "activity": activity.detach().cpu().item(),
                    "bump_term": bump_term.detach().cpu().item(),
                    "const_term": constraint_term.detach().cpu().item(),
                    **{
                        c: log_functions[c](vec).detach().cpu().item()
                        for c in log_functions
                    },
                }
            )

            v1 = history[-1]["emb"]
            v2 = history[-2]["emb"]
            s1 = history[-1]["smiles"]
            s2 = history[-2]["smiles"]

            # build log string
            log_str = f"{k}: dV {np.linalg.norm(v1-v2):.3e} "
            to_log = ["loss", "activity", "bump_term", "const_term"] + list(
                log_functions.keys()
            )
            for l in to_log:
                log_str = log_str + f"{l}:{history[-1][l]:.2f} "

            if (
                ((v1 * v2).sum() / (np.linalg.norm(v1) * np.linalg.norm(v2)) > 0.85)
                and (k - last_bump > 25)
            ) or (s1 == s2 and k > 50):
                print("adding bump ", smiles)
                last_bump = k
                new_bump = torch.from_numpy(v1).to(device=vec.device)
                bumps.append(new_bump)

            if save_traj_history is not None:
                # save trajectory to file
                with open(save_traj_history, "wb") as f:
                    pickle.dump(history, f)

            print(log_str)

        optimizer.step()
    return history

def embed_and_score_in_batches(
    records,
    encoder,
    tokenizer,
    batch_size=128,
    rdkit_scores={},
    model_scores={},
    smiles_field="smiles",
):
    # A helper function to compute embeddings from the model encoder and
    # rdkit properties of molecules in batches. Predictions from other models
    # can also be supplied.
    #
    # The input list of dict records is modified in place.

    # check that the rdkit and model scores are functions
    assert all([callable(score_fn) for score_fn in rdkit_scores.values()])

    # check that the rdkit and model keys are unique
    assert len(set(rdkit_scores.keys()).intersection(set(model_scores.keys()))) == 0

    print("Embedding and scoring iterable from smiles.")
    batch_iter = batch_indexable(records, batch_size)
    num_batches = len(records) // batch_size
    with torch.no_grad():
        for i, batch in enumerate(batch_iter):
            print(f"batch: {i}/{num_batches}")
            try:
                batch_mols = [Chem.MolFromSmiles(row[smiles_field]) for row in batch]
                batch_smiles = [Chem.MolToSmiles(m) for m in batch_mols]
                batch_tokens = torch.tensor(
                    [
                        tokenizer.tokenize_text("[SMILES]" + s + "[STOP]", pad=True)
                        if s != "*"
                        else tokenizer.tokenize_text("[SMILES]C[STOP]", pad=True)
                        for s in batch_smiles
                    ],
                    device=encoder.device,
                    dtype=torch.int,
                )
                batch_embeds = encoder.encode_tokens(batch_tokens, tokenizer)

                batch_scores = {}
                for quantity, score_fn in rdkit_scores.items():
                    batch_scores[quantity] = [score_fn(m) for m in batch_mols]
                for quantity, score_fn in model_scores.items():
                    batch_scores[quantity] = score_fn(batch_embeds).squeeze()

                for k, r in enumerate(batch):
                    batch[k]["emb_smiles"] = batch_embeds[k].detach().cpu().numpy()
                    for quantity in batch_scores.keys():
                        if quantity in model_scores.keys():
                            batch[k][quantity] = (
                                batch_scores[quantity][k].detach().cpu().item()
                            )
                        else:
                            batch[k][quantity] = batch_scores[quantity][k]

            except Exception as Ex:
                print(Ex)
                continue

def get_qed(v):
    return qed_model(v.unsqueeze(0)).mean

def get_logp(v):
    return logp_model(v.unsqueeze(0)).mean

def logp_penalty(v):
    """
    Penalize logP > 5. Loss is squared to make it smooth.
    """
    return torch.pow(4 * torch.nn.functional.relu(get_logp(v) - 5.0), 2.0)

def metadynamics(args):
    global qed_model, logp_model
    DEVICE = torch.device(args.device)
    encoder, tokenizer = load_e3gnn_smiles_clip_e2e(
        freeze=True,
        device=DEVICE,
        # model parameters to load.
        doc_url=args.model,
    )

    with open(args.smiles) as f:
        smiles = [line.strip('\n') for line in f]
    init_mols = [{"smiles": s} for s in smiles[:3]]

    subset = [{"smiles": s, "source": "demo_mols"} for s in smiles[:1000]] # only for load model
    embed_and_score_in_batches_regression(subset, encoder, tokenizer)
    qed_model, _ = basic_due(
        subset,
        x_field="emb_smiles",
        y_field="qed",
        load_as="regression_qed.pkl",
        continue_training=False,
        random_seed=args.seed
        )
    logp_model, _ = basic_due(
        subset,
        x_field="emb_smiles",
        y_field="logp",
        load_as="regression_logp.pkl",
        continue_training=False,
        random_seed=args.seed
        )

    nsteps = 2000
    meta_traj_no_binding = []
    for k, rec in enumerate(init_mols):
        traj = coati_metadynamics(
            embed_smiles(rec["smiles"], encoder, tokenizer).to(DEVICE),  
            lambda X: -10 * get_qed(X),  # the constraints below will be added to this.
            encoder,
            tokenizer,
            bump_radius=0.125 * 4,
            bump_height=80.0 * 16,
            constraint_functions={
                "logp_p": logp_penalty
            },  # enforced to == 0 by lagrange multipliers.
            log_functions={
                "logp": get_logp,
                "qed": get_qed,
            },  # These log functions will get appended to the history list-dict returned.
            nsteps=nsteps,
            save_traj_history=f"./meta_traj_no_binding_{k}.pkl",
        )
        for r in traj:
            r["library"] = "opt_" + str(k)
        meta_traj_no_binding.extend(traj)

    rdkit_scores = {
        "qed_rdkit": rdkit.Chem.QED.qed,
        "logp_rdkit": rdkit.Chem.Crippen.MolLogP,
        "mol_wt_rdkit": rdkit.Chem.Descriptors.ExactMolWt,
    }
    model_scores = {"qed_model": get_qed, "logp_model": get_logp}
    embed_and_score_in_batches(
        meta_traj_no_binding, encoder, tokenizer, 
        rdkit_scores=rdkit_scores, model_scores=model_scores
    )

    # save 1
    for r in meta_traj_no_binding:
        r["library"] = "meta_qed"
    to_score = meta_traj_no_binding 
    df = pd.DataFrame(to_score)
    embs = np.stack(df["emb_smiles"].values.tolist(), 0)
    X_embedded = TSNE(n_components=2, learning_rate=100, init="random").fit_transform(embs)
    df.loc[:, "X"] = X_embedded[:, 0]
    df.loc[:, "Y"] = X_embedded[:, 1]
    norm = matplotlib.colors.Normalize(vmin=0, vmax=5.0)
    plt.figure(figsize=(10, 8))
    plt.scatter(
        df[df.library == "meta_qed"].X,
        df[df.library == "meta_qed"].Y,
        #c=df[df.library == "meta_qed"].pic50_model,
        norm=norm,
        marker="s",
        cmap="plasma",
        alpha=0.5,
        label="no_bind_traj",
    )
    plt.clim(0, 10.0)
    plt.legend()
    plt.colorbar()
    plt.axis("off")
    plt.savefig("trajs_tsne.pdf", dpi=300, bbox_inches="tight")
    
    # save 2
    chart = embed_altair(
    df,
    tooltip_fields=["smiles", "mol_index", "logp_rdkit", "qed_rdkit"],
    selector_field="library",
    quantity="qed_rdkit",
    image_tooltip=True,
    emb_field="emb_smiles",
    smiles_field="smiles",
    width=1024,
    height=768,
    )
    chart.save("meta_dyn_trajs.html")
    # save 3
    images = []
    for traj in meta_traj_no_binding:
        try:
            k = traj["name"]
            qed = traj["qed_model"]
            logp = traj["logp_model"]
            images.append(
                Chem.Draw.MolToImage(
                    Chem.MolFromSmiles(traj["smiles"]),
                    legend=f"k {k} qed {qed:.2f} logp {logp:.2f}",
                )
            )
        except:
            pass
    images[0].save(
        "qed_meta_example_2k_steps.gif",
        save_all=True,
        append_images=images[1:],
        optimize=False,
        duration=120,
        loop=0,
    )

if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--mode', choices=['basic', 'near', 'regression', 'dynamics'], \
        default='basic',help='Generation mode')
    arg_parser.add_argument('--device', choices=['cuda:0', 'cpu'], \
        default='cuda:0',help='Device')
    arg_parser.add_argument('--seed', type=int, default=2024) 
    arg_parser.add_argument('--model', type=str, default = 'models/ecloud_augmented_37.pkl')
    arg_parser.add_argument('--dataset', type=str, default='data/ecloud_coati_demo.pt')
    arg_parser.add_argument('--smiles', type=str, default='data/demo.smi')
    arg_parser.add_argument('--cached_data', type=str, default=None)
    arg_parser.add_argument('--output', type=str, default='0423_near_mol4_30.txt')
    arg_parser.add_argument('--noise', type=float, default=0.3)
    args = arg_parser.parse_args()
    setup_seed(args.seed)
    print('Mode: ' + args.mode)

    if args.mode == 'basic':
        basic_generation(args)
    elif args.mode == 'near':
        generation_near_a_given_mol(args)
    elif args.mode == 'regression':
        train_regression(args, y_field="logp")
    elif args.mode == 'dynamics':
        metadynamics(args)