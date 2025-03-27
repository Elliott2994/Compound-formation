import streamlit as st
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, DataStructs
from mol2vec.features import mol2alt_sentence, MolSentence, DfVec, sentences2vec
from gensim.models import word2vec
import pandas as pd
from io import BytesIO
from tqdm import tqdm
import tempfile
from transformers import AutoTokenizer, AutoModelForMaskedLM, BartTokenizer, BartModel
import torch

# 读取Excel文件
def read_smiles_from_excel(file_path, sheet_name=0, column_index=0):
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    smiles_list = df.iloc[:, column_index].dropna().tolist()
    return smiles_list

# 读取TXT文件
def read_smiles_from_txt(file_path):
    with open(file_path, 'r') as file:
        smiles_list = [line.strip() for line in file.readlines() if line.strip()]
    return smiles_list

# 计算ECFP指纹
def smiles_to_ecfp(smiles_list, radius=2, nBits=1024):
    ecfp_list = []
    for smiles in tqdm(smiles_list, desc="Computing ECFP"):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            ecfp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
            ecfp_list.append(ecfp)
        else:
            st.write(f"Invalid SMILES: {smiles}")
    return np.array(ecfp_list)

# 计算MACCS指纹
def smiles_to_maccs(smiles_list):
    maccs_list = []
    for smiles in tqdm(smiles_list, desc="Computing MACCS"):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            maccs_fp = MACCSkeys.GenMACCSKeys(mol)
            maccs_list.append(maccs_fp)
        else:
            st.write(f"Invalid SMILES: {smiles}")
    return np.array(maccs_list)

# 计算ExtFP指纹
def smiles_to_extfp(smiles_list, radius=2, nBits=1024):
    extfp_list = []
    for smiles in tqdm(smiles_list, desc="Computing ExtFP"):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
            arr = np.zeros((1,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(fp, arr)
            extfp_list.append(arr)
        else:
            st.write(f"Invalid SMILES: {smiles}")
    return np.array(extfp_list)

# 计算Mol2Vec表示
def smiles_to_mol2vec(smiles_list, vector_size=100, window=5, min_count=1, workers=4):
    mols = [Chem.MolFromSmiles(s) for s in smiles_list if Chem.MolFromSmiles(s) is not None]
    mol_sentences = [MolSentence(mol2alt_sentence(mol, radius=1)) for mol in mols]

    model = word2vec.Word2Vec(mol_sentences, vector_size=vector_size, window=window, min_count=min_count, workers=workers)

    vectors = sentences2vec(mol_sentences, model, unseen='UNK')
    return np.array(vectors)

# 计算ChemBERTa表示
def smiles_to_chemberta(smiles_list):
    tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MLM")
    model = AutoModelForMaskedLM.from_pretrained("DeepChem/ChemBERTa-77M-MLM")

    representations = []
    for smiles in tqdm(smiles_list, desc="Computing ChemBERTa"):
        inputs = tokenizer(smiles, return_tensors="pt", truncation=True, padding='max_length', max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        averaged_representation = logits.mean(dim=1).cpu().numpy()
        representations.append(averaged_representation)
    return np.array(representations)

# 计算MegaMolBERT表示
def smiles_to_megamolbert(smiles_list):
    model_name = "C:\\Users\\86151\\Desktop\\大创\\human、celegans数据去重\\BERT_Model"
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartModel.from_pretrained(model_name)

    representations = []
    for smiles in tqdm(smiles_list, desc="Computing MegaMolBERT"):
        inputs = tokenizer(smiles, return_tensors="pt", truncation=True, padding="max_length", max_length=1024)
        with torch.no_grad():
            outputs = model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        molecular_representation = last_hidden_states[:, 0, :].cpu().numpy()
        representations.append(molecular_representation)
    return np.array(representations)

# Streamlit 应用
def main():
    st.title("Molecular Fingerprint Generator")

    # 文件上传
    uploaded_file = st.file_uploader("Upload a file", type=["xlsx", "xls", "txt"], key="file_uploader_1")
    if uploaded_file is not None:
        # 读取文件
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        # 根据文件类型读取SMILES列表
        if uploaded_file.name.endswith('.txt'):
            smiles_list = read_smiles_from_txt(tmp_file_path)
        else:
            smiles_list = read_smiles_from_excel(tmp_file_path, sheet_name=0, column_index=0)

        # 选择指纹类型
        fingerprint_type = st.selectbox("Select fingerprint type", ["ECFP", "MACCS", "ExtFP", "Mol2Vec", "ChemBERTa", "MegaMolBERT"])

        # 计算指纹
        if st.button("Generate Fingerprint"):
            if fingerprint_type == "ECFP":
                fingerprints = smiles_to_ecfp(smiles_list)
            elif fingerprint_type == "MACCS":
                fingerprints = smiles_to_maccs(smiles_list)
            elif fingerprint_type == "ExtFP":
                fingerprints = smiles_to_extfp(smiles_list)
            elif fingerprint_type == "Mol2Vec":
                fingerprints = smiles_to_mol2vec(smiles_list)
            elif fingerprint_type == "ChemBERTa":
                fingerprints = smiles_to_chemberta(smiles_list)
            elif fingerprint_type == "MegaMolBERT":
                fingerprints = smiles_to_megamolbert(smiles_list)

            # 下载指纹
            buffer = BytesIO()
            np.save(buffer, fingerprints)
            st.download_button(
                label="Download Fingerprint",
                data=buffer.getvalue(),
                file_name=f"{fingerprint_type}_fingerprints.npy",
                mime="application/octet-stream"
            )

if __name__ == "__main__":
    main()
