from rdkit import Chem
import numpy as np
from datetime import datetime
import pickle
import pandas as pd
import time
import json
import sys
import util
import netutil
import warnings
import torch
from urllib.parse import urlparse
import io
import gzip
import predwrap
import os
from io import StringIO

warnings.filterwarnings("ignore")

nuc_to_atomicno = {"13C": 6, "1H": 1}


def predict_mols(
    raw_mols,
    predictor,
    MAX_N,
    to_pred=None,
    add_h=True,
    sanitize=True,
    add_default_conf=True,
    num_workers=0,
):
    t1 = time.time()
    if add_h:
        mols = [Chem.AddHs(m) for m in raw_mols]
    else:
        mols = [Chem.Mol(m) for m in raw_mols]  # copy

    if sanitize:
        [Chem.SanitizeMol(m) for m in mols]

    # sanity check
    for m in mols:
        if m.GetNumAtoms() > MAX_N:
            raise ValueError("molecule has too many atoms")

        if len(m.GetConformers()) == 0 and add_default_conf:
            print("adding default conf")
            util.add_empty_conf(m)

    if to_pred in ["13C", "1H"]:
        pred_fields = ["pred_shift_mu", "pred_shift_std"]
    else:
        raise ValueError(f"Don't know how to predict {to_pred}")

    pred_t1 = time.time()
    vert_result_df, edge_results_df = predictor.pred(
        [{"rdmol": m} for m in mols],
        pred_fields=pred_fields,
        BATCH_SIZE=256,
        num_workers=num_workers,
    )

    pred_t2 = time.time()
    # print("The prediction took {:3.2f} ms".format((pred_t2-pred_t1)*1000)),

    t2 = time.time()

    all_out_dict = []

    ### pred cleanup
    if to_pred in ["13C", "1H"]:
        shifts_df = pd.pivot_table(
            vert_result_df,
            index=["rec_idx", "atom_idx"],
            columns=["field"],
            values=["val"],
        ).reset_index()

        for rec_idx, mol_vert_result in shifts_df.groupby("rec_idx"):
            m = mols[rec_idx]
            out_dict = {"smiles": Chem.MolToSmiles(m)}

            # tgt_idx = [int(a.GetIdx()) for a in m.GetAtoms() if a.GetAtomicNum() == nuc_to_atomicno[to_pred]]

            # a = mol_vert_result.to_dict('records')
            out_shifts = []
            # for row_i, row in mol_vert_result.iterrows():
            for row in mol_vert_result.to_dict(
                "records"
            ):  # mol_vert_result.iterrows():
                atom_idx = int(row[("atom_idx", "")])
                if (
                    m.GetAtomWithIdx(atom_idx).GetAtomicNum()
                    == nuc_to_atomicno[to_pred]
                ):
                    out_shifts.append(
                        {
                            "atom_idx": atom_idx,
                            "pred_mu": row[("val", "pred_shift_mu")],
                            "pred_std": row[("val", "pred_shift_std")],
                        }
                    )

            out_dict[f"shifts_{to_pred}"] = out_shifts

            out_dict["success"] = True
            all_out_dict.append(out_dict)

    return all_out_dict


DEFAULT_FILES = {
    "13C": {
        "meta": "models/default_13C.meta",
        "checkpoint": "models/default_13C.checkpoint",
    },
    "1H": {
        "meta": "models/default_1H.meta",
        "checkpoint": "models/default_1H.checkpoint",
    },
}


def s3_split(url):
    o = urlparse(url)
    bucket = o.netloc
    key = o.path.lstrip("/")
    return bucket, key

#@click.option('--filename', help='filename of file to read, or stdin if unspecified', default=None)
#@click.option('--format', help='file format (sdf, rdkit)', default='sdf', 
#              type=click.Choice(['rdkit', 'sdf'], case_sensitive=False))
#@click.option('--pred', help='Nucleus (1H or 13C) or coupling (coupling)', default='13C', 
#              type=click.Choice(['1H', '13C', 'coupling'], case_sensitive=True))
#@click.option('--model_meta_filename')
#@click.option('--model_checkpoint_filename')
#@click.option('--print_data', default=None, help='print the smiles/fingerprint of the data used for train or test') 
#@click.option('--output', default=None)
#@click.option('--num_data_workers', default=0, type=click.INT)
#@click.option('--cuda/--no-cuda', default=True)
#@click.option("--version", default=False, is_flag=True)
#@click.option('--sanitize/--no-sanitize', help="sanitize the input molecules", default=True)
#@click.option('--addhs', help="Add Hs to the input molecules", default=False)
#@click.option('--skip-molecule-errors/--no-skip-molecule-errors', help="skip any errors", default=True)
def predict(filecontent, format, pred, model_meta_filename=None, 
            model_checkpoint_filename=None, cuda=False, 
            output=None, sanitize=True, addhs=True,
            print_data = None, version=False,
            skip_molecule_errors=True, num_data_workers=0):
    ts_start = time.time()
    if version:
        print(os.environ.get("GIT_COMMIT", ""))
        sys.exit(0)

    if model_meta_filename is None:
        # defaults
        model_meta_filename = DEFAULT_FILES[pred]["meta"]
        model_checkpoint_filename = DEFAULT_FILES[pred]["checkpoint"]

    if print_data is not None:
        data_info_filename = model_meta_filename.replace(
            ".meta", "." + print_data + ".json"
        )
        print(open(data_info_filename, "r").read())
        sys.exit(0)

    meta = pickle.load(open(model_meta_filename, "rb"))

    MAX_N = meta["max_n"]

    cuda_attempted = cuda
    if cuda and not torch.cuda.is_available():
        warnings.warn("CUDA requested but not available, running with CPU")
        cuda = False
    predictor = predwrap.PredModel(
        model_meta_filename,
        model_checkpoint_filename,
        cuda,
        override_pred_config={},
    )

    input_fileobj = None

    if format == 'sdf':
        mol_supplier = Chem.SDMolSupplier()
        mol_supplier.SetData(filecontent)
    elif format == 'rdkit':
        if filename is None:
            bin_data = sys.stdin.buffer.read()
            mol_supplier = [Chem.Mol(m) for m in pickle.loads(bin_data)]
        elif input_fileobj is not None:
            mol_supplier = [Chem.Mol(m) for m in pickle.load(input_fileobj)]
        else:
            mol_supplier = [Chem.Mol(m) for m in pickle.load(open(filename, "rb"))]

    mols = list(mol_supplier)
    if len(mols) > 0:
        all_results = predict_mols(
            mols,
            predictor,
            MAX_N,
            pred,
            add_h=addhs,
            sanitize=sanitize,
            num_workers=num_data_workers,
        )
    else:
        all_results = []
    ts_end = time.time()
    output_dict = {
        "predictions": all_results,
        "meta": {
            "max_n": MAX_N,
            "to_pred": pred,
            "model_checkpoint_filename": model_checkpoint_filename,
            "model_meta_filename": model_meta_filename,
            "ts_start": datetime.fromtimestamp(ts_start).isoformat(),
            "ts_end": datetime.fromtimestamp(ts_end).isoformat(),
            "runtime_sec": ts_end - ts_start,
            "git_commit": os.environ.get("GIT_COMMIT", ""),
            "rate_mol_sec": len(all_results) / (ts_end - ts_start),
            "num_mol": len(all_results),
            "cuda_attempted": cuda_attempted,
            "use_cuda": cuda,
        },
    }
    json_str = json.dumps(output_dict, sort_keys=False, indent=4)
    if output is None:
        print(json_str)
    else:
        if output.startswith("s3://"):
            bucket, key = s3_split(output)
            s3 = boto3.client("s3")

            json_bytes = json_str.encode("utf-8")
            if key.endswith(".gz"):
                json_bytes = gzip.compress(json_bytes)

            output_fileobj = io.BytesIO(json_bytes)
            s3.upload_fileobj(output_fileobj, bucket, key)

        else:
            with open(output, "w") as fid:
                fid.write(json_str)


if __name__ == "__main__":
    predict("\n  CDK     09072315522D\nnmrshiftdb2 60015778\n 27 28  0  0  0  0  0  0  0  0999 V2000\n   -1.3616    0.8027    0.0000 C   0  0  0  0  0  3  0  0  0  0  0  0\n   -2.0760    0.3902    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n   -2.0760   -0.4348    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n   -1.3616   -0.8473    0.0000 C   0  0  0  0  0  3  0  0  0  0  0  0\n   -0.6471   -0.4348    0.0000 C   0  0  0  0  0  3  0  0  0  0  0  0\n   -0.6471    0.3902    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n    0.0673    0.8027    0.0000 C   0  0  0  0  0  3  0  0  0  0  0  0\n    0.0673    1.6277    0.0000 C   0  0  0  0  0  3  0  0  0  0  0  0\n   -0.6471    2.0402    0.0000 C   0  0  0  0  0  2  0  0  0  0  0  0\n    0.7816    2.0402    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n    1.4961    1.6277    0.0000 C   0  0  0  0  0  3  0  0  0  0  0  0\n    2.2107    2.0402    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n    2.2108    2.8652    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n    1.4963    3.2777    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n    0.7816    2.8652    0.0000 C   0  0  0  0  0  3  0  0  0  0  0  0\n   -1.3616    1.6277    0.0000 O   0  0  0  0  0  1  0  0  0  0  0  0\n    0.7816    0.3902    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0\n    0.7816   -0.4347    0.0000 C   0  0  0  0  0  2  0  0  0  0  0  0\n    0.0673   -0.8472    0.0000 C   0  0  0  0  0  1  0  0  0  0  0  0\n   -2.7905    0.8027    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0\n   -2.7905   -0.8473    0.0000 O   0  0  0  0  0  1  0  0  0  0  0  0\n   -3.5049    0.3901    0.0000 C   0  0  0  0  0  1  0  0  0  0  0  0\n    1.4963    4.1026    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0\n    0.7819    4.5153    0.0000 C   0  0  0  0  0  1  0  0  0  0  0  0\n    2.9253    3.2777    0.0000 O   0  0  0  0  0  1  0  0  0  0  0  0\n    2.9251    1.6277    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0\n    2.9251    0.8027    0.0000 C   0  0  0  0  0  1  0  0  0  0  0  0\n  1  2  2  0  0  0  0 \n  2  3  1  0  0  0  0 \n  3  4  2  0  0  0  0 \n  4  5  1  0  0  0  0 \n  5  6  2  0  0  0  0 \n  6  1  1  0  0  0  0 \n  6  7  1  0  0  0  0 \n  7  8  1  0  0  0  0 \n  8  9  1  1  0  0  0 \n  8 10  1  0  0  0  0 \n 11 12  1  0  0  0  0 \n 12 13  2  0  0  0  0 \n 13 14  1  0  0  0  0 \n 14 15  2  0  0  0  0 \n 10 11  2  0  0  0  0 \n 15 10  1  0  0  0  0 \n 18 19  1  0  0  0  0 \n 12 26  1  0  0  0  0 \n 26 27  1  0  0  0  0 \n 14 23  1  0  0  0  0 \n 23 24  1  0  0  0  0 \n 13 25  1  0  0  0  0 \n  9 16  1  0  0  0  0 \n  7 17  1  1  0  0  0 \n 17 18  1  0  0  0  0 \n  3 21  1  0  0  0  0 \n  2 20  1  0  0  0  0 \n 20 22  1  0  0  0  0 \nM  END\n\n","sdf","13C")
