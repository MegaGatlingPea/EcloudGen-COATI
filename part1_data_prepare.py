import h5py
from rdkit import Chem
from coati.models.encoding.tokenizers.trie_tokenizer import TrieTokenizer
from coati.models.encoding.tokenizers import get_vocab
from tqdm import tqdm

f = h5py.File('data/ecloud.h5', 'r')
ecloud_item = f['eclouds']
with open('data/all.smi') as f:
        smiles=[line.strip('\n') for line in f]
print('data len: ', len(smiles))

tokenizer = TrieTokenizer(n_seq=64, **get_vocab('mar'))
with h5py.File('data/ecloud_coati.h5','w') as f_:
        eclouds=f_.create_dataset("eclouds", (len(smiles),32,32,32), dtype='f')
        raw_tokens=f_.create_dataset("raw_tokens", (len(smiles),64), dtype='i')
        augmented_tokens=f_.create_dataset("augmented_tokens", (len(smiles),64), dtype='i')
        for i, s in tqdm(enumerate(smiles)):
                s = Chem.MolToSmiles(Chem.MolFromSmiles(s))
                raw_token = tokenizer.tokenize_text("[SMILES]" + s + "[STOP]", pad=True)
                augmented_token = tokenizer.tokenize_text("[CLIP][UNK][SMILES][SUFFIX][MIDDLE]" + s + "[STOP]", pad=True)
                eclouds[i] = ecloud_item[i]
                raw_tokens[i] = raw_token
                augmented_tokens[i] = augmented_token
                if i % 100000 == 0:
                        print('process', i)

f.close()


