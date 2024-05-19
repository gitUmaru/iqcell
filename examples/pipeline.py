# pip install iqcell
from iqcell.preprocessing import select_highly_variable_genes, correct_dropout
from iqcell.network_generation import generate_interaction_network
from iqcell.binarization import binarize_expression
from iqcell.hierarchy_generation import generate_gene_hierarchy
from iqcell.reasoning_engine import implement_reasoning_engine
import iqcell.utils

scRNA_data = iqcell.utils.readdata(
    expression='examples/data/espression.csv',
    pseudotime="examples/data/pseudotime.csv"
)
highly_variable_genes = select_highly_variable_genes(scRNA_data)
corrected_data = correct_dropout(scRNA_data)
interaction_network = generate_interaction_network(corrected_data)
binarized_data = binarize_expression(corrected_data)
gene_hierarchy = generate_gene_hierarchy(interaction_network, binarized_data)
predictions = implement_reasoning_engine(interaction_network, gene_hierarchy)
