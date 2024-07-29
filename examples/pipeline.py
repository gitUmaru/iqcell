# pip install iqcell==2.*
import iqcell
import scanpy as sc

# Load scRNA-seq data
scRNA_data = sc.read('scRNA_data.h5ad')

# Preprocess data using non-iqcell packages (out of scope for iqcell)
highly_variable_genes = iqcell.preprocessing.select_highly_variable_genes(scRNA_data) # use pyScenic
scRNA_data = iqcell.utils.select(genes=highly_variable_genes, data=scRNA_data)

corrected_data = iqcell.preprocessing.correct_dropout(scRNA_data) # use MAGIC

# Binarize expression data
binarizer = iqcell.binarization.KMeans()
binarized_data = binarizer.discretize(corrected_data)

# Implement reasoning engine
z3 = iqcell.logic_engine.Z3(data=binarized_data)
predictions_z3 = z3.predict()

binn = iqcell.logic_engine.BINN(data=binarized_data)
predictions_binn = binn.predict()

# Output predictions
print("Z3 Predictions:", predictions_z3)
print("BINN Predictions:", predictions_binn)