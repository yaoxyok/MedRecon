# %%
import numpy as np
import re
import pandas as pd
import py2neo
from optimizer import *
import pickle
from ast import literal_eval
from collections import defaultdict

chron_mapping = pd.read_csv('data/chron.csv')
antecedent_mapping = pd.read_csv('data/antecedent.csv')
antecedent_mapping['mapping']=antecedent_mapping.apply(lambda x: [x.icd[7:],x['value'][7:]] if not pd.isna(x['value']) else [x.icd[7:]], axis = 1)
chron_mapping['mapping']=chron_mapping.apply(lambda x: [x.icd[7:],x['value'][7:]] if not pd.isna(x['value']) else [x.icd[7:]], axis = 1)
antecedent_mapping = antecedent_mapping.set_index('antecedent')['mapping'].to_dict()
chron_mapping = chron_mapping.set_index('chron')['mapping'].to_dict()

# %%
def discrepancy_detector(query_row,neo4j_graph):
    drug_cur_codes = query_row.atcs
    disease_cur_codes = query_row.icd9
    disease_chron_codes = [] if pd.isna(query_row.chron) else set().union(*[chron_mapping[each] for each in query_row.chron[:-1].split("', '")])
    disease_ante_codes = [] if pd.isna(query_row.antecedent) else set().union(*[antecedent_mapping[each] for each in query_row.antecedent[:-1].split("', '")])
    disease_codes = set().union(*[disease_cur_codes,disease_chron_codes,disease_ante_codes])

    res = []
    ##### Contraindication
    query = f"""
        WITH {'["'+ '","'.join(disease_codes) +'"]'} AS atcs, {'["'+ '","'.join(disease_cur_codes) +'"]'} AS icd9s
        UNWIND atcs AS atc
        UNWIND icd9s AS icd9
        WITH atc, icd9
        MATCH (drug:DrugProfile)-[r:HAS_CONTRAIND]->(parent:DiseaseProfile)<-[:HAS_PARENTCODE*]-(disease:DiseaseProfile)
        WHERE atc IN drug.atcs and icd9 IN disease.icd9cms
        RETURN drug.atcs AS drug, disease.icd9cms AS disease
    """
    contraindication = neo4j_graph.run(query).to_data_frame().values
    if len(contraindication) > 0:
        res.append(['Over-contraindication', contraindication])
    
    ##### Interactions
    query = f"""
        WITH {'["'+ '","'.join(drug_cur_codes) +'"]'} AS atcs
        UNWIND atcs AS drugA
        UNWIND atcs AS drugB
        WITH drugA, drugB WHERE drugA < drugB
        MATCH (d1:DrugProfile)-[r:INTERACTS]-(d2:DrugProfile)
        WHERE drugA IN d1.atcs and drugB IN d2.atcs 
        RETURN d1.name AS Drug1, d2.name AS Drug2
    """
    
    interaction = neo4j_graph.run(query).to_data_frame().values
    if len(interaction) > 0:
        res.append(['Over-Interactions', interaction])


    ###### Under
    for disease_code in disease_cur_codes:
        
        query = f"""
            MATCH (drug:DrugProfile)-[r:TREATS]->(ds:DiseaseProfile)
            WHERE '{disease_code}' IN ds.icd9cms 
            RETURN drug.name AS Drug, ds.name AS Disease, type(r) AS Relationship
        """
        potential_treat = neo4j_graph.run(query).to_data_frame().values

        if len(potential_treat) != 0:
            query = f"""
                WITH {'["'+ '","'.join(drug_cur_codes) +'"]'} AS atcs, {'"'+disease_code+'"'}  AS icd
                UNWIND atcs AS atc
                WITH atc, icd
                MATCH (drug:DrugProfile)-[r:TREATS]->(ds:DiseaseProfile)
                WHERE atc IN drug.atcs AND icd IN ds.icd9cms 
                RETURN drug.name AS Drug, ds.name AS Disease, type(r) AS Relationship
            """
            treat = neo4j_graph.run(query).to_data_frame().values
            if len(treat) == 0:
                res.append(['Under', disease_code])
        else:
            return 'Exclude'
    if not res:
        return 'Appropriate'
    else:
        return res

# %%

# %%
def generate_drug_candidate(query_row,neo4j_graph, k = 4):

    print('start generate drug candidate')
    drug_cur_codes = query_row.atcs
    drug_cur_cat = [code[:k] for code in drug_cur_codes]
    icd9codes = query_row.icd9
    print('icd9codes',icd9codes)
    if isinstance(query_row.antecedent, list):
        for each in query_row.antecedent:
            icd9codes.extend(antecedent_mapping[each])
    if isinstance(query_row.chron, list):
        for each in query_row.chron:
            icd9codes.extend(chron_mapping[each])
    print('icd9codes',icd9codes)
    icd9codes = list(set(icd9codes))
    drug_list = []
    drug_cat_list = []
    disease_list = []
    drug_cur_list = []
    for code in icd9codes:
        query = f"""
            MATCH (d:DiseaseProfile)-[r:TREATS]-(drug:DrugProfile)
            WHERE 'ICD9CM/{code}' in d.icd9cms 
            RETURN drug.drugbank_id as drug, drug.atcs as atcs, d.id as disease
        """
        df_temp = neo4j_graph.run(query).to_data_frame()
        cat_pairs = set()

        if len(df_temp) > 0:
            for row in df_temp.iterrows():
                if len(row[1].atcs) > 0:
                    cat = row[1].atcs[0][:k]
                    for atc in row[1].atcs:
                        if atc[:k] in drug_cur_cat:
                            cat = atc[:k]
                            cat_pairs.add((code, cat))
                            break
                    drug_cat_list.append(cat)
                    drug_list.append(row[1].drug)
                    disease_list.append(code)

    treat_pairs = set(zip(drug_list,disease_list, drug_cat_list))
    treat_level = defaultdict(int)
    for disease, cat in cat_pairs:
        treat_level[disease] += 1


    for atc in drug_cur_codes:
        query = f"""
            MATCH (drug:DrugProfile)
            WHERE '{atc}' IN drug.atcs 
            RETURN drug.drugbank_id as drug
        """
        df_temp = neo4j_graph.run(query).to_data_frame()
        if len(df_temp) > 0:
            drug_cur_list.extend(list(df_temp.drug.values))
    
    drug_cur_set = set(drug_cur_list)
    full_drug_set = drug_cur_set.union(set(drug_list))
    drug_new_candidate = full_drug_set - drug_cur_set
    
    return full_drug_set, drug_cur_set, drug_new_candidate, treat_pairs, set(disease_list), treat_level, set(drug_cat_list)

def overlap_check(drug_cur_set, disease_set, neo4j_graph):
    query = f"""
        MATCH (drug:DrugProfile)-[r:TREATS]-(disease:DiseaseProfile)
        WHERE drug.drugbank_id IN {list(drug_cur_set)} and disease.id IN {list(disease_set)}
        RETURN drug.drugbank_id AS drug, disease.id AS disease
    """
    df_temp = neo4j_graph.run(query).to_data_frame()
    if len(df_temp) > 0:
        return True
    else:
        return False
# %%
def generate_DDI_matrix(D,idx_to_drug,neo4j_graph):

    print('start generate DDI matrix')

    DDI_matrix = np.zeros((D,D))
    for d1 in range(D-1):
        for d2 in range(d1+1,D):
            drug1, drug2 = idx_to_drug[d1], idx_to_drug[d2]
            query = f"""
                MATCH (d1:DrugProfile)-[r:INTERACTS]-(d2:DrugProfile)
                WHERE d1.name = "{drug1}" and d2.name = "{drug2}"
                RETURN d1,d2
            """
            if len(neo4j_graph.run(query).to_data_frame())>0:
                DDI_matrix[d1][d2] = 1
                DDI_matrix[d2][d1] = 1
    return DDI_matrix

# %%
def generate_contrain_matrix(D,C,idx_to_drug,idx_to_disease,neo4j_graph):

    print('start generate Contraindication matrix')
    contra_matrix = np.zeros((D,C))
    for d in range(D):
        for c in range(C):
            drug, disease = idx_to_drug[d], idx_to_disease[c]
            query = f"""
                MATCH (drug:DrugProfile)-[r:HAS_CONTRAIND]-(disease:DiseaseProfile)
                WHERE drug.name = "{drug}" and disease.name = "{disease}"
                RETURN drug.name AS drug, disease.name AS disease
            """
            if len(neo4j_graph.run(query).to_data_frame())>0:
                contra_matrix[d][c] = 1
    return contra_matrix


def reconciled_list_generator(query_row, best_embed,neo4j_graph):
    full_drug_set, drug_cur_set, drug_new_candidate, pairs, disease_set, treat_level, cat_set = generate_drug_candidate(query_row,neo4j_graph)
    print('disease_set',disease_set)
    print('drug_cur_set',drug_cur_set)
    # %%
    drug_to_idx = {each:i for i,each in enumerate(full_drug_set)}
    idx_to_drug = {i:each for i,each in enumerate(full_drug_set)}
    # print('drug_to_idx',drug_to_idx)
    disease_to_idx = {each:i for i,each in enumerate(disease_set)}
    idx_to_disease = {i:each for i,each in enumerate(disease_set)}

    cat_to_idx = {each:i for i,each in enumerate(cat_set)}
    idx_to_cat = {i:each for i,each in enumerate(cat_set)}

    D = len(full_drug_set)
    C = len(disease_set)
    K = len(cat_set)
    Dc = [drug_to_idx[each] for each in drug_cur_set]
    Dp = [drug_to_idx[each] for each in drug_new_candidate]
    # print('D:',D,'C:',C,'Dc len:',len(Dc),'Dp len:',len(Dp))
    ### curative target t
    t = np.ones(C)
    for disease, level in treat_level.items():
        if disease in disease_set:
            t[disease_to_idx[disease]] = level
    # print('t',t)
    ### treament matrix E
    E = np.zeros((D,C))
    Dk = defaultdict(set)
    for drug, disease, cat in pairs:
        E[drug_to_idx[drug]][disease_to_idx[disease]] = 1
        Dk[cat_to_idx[cat]].add(drug_to_idx[drug])
    ### Contraindication matrix N
    N = generate_contrain_matrix(D,C,idx_to_drug,idx_to_disease,neo4j_graph)
    ### Interaction matrix I
    I = generate_DDI_matrix(D,idx_to_drug,neo4j_graph)
    ### Embedding m
    with open('data/node_name_map.pkl','rb') as f:
        node_name_map = pickle.load(f)
    # print(best_embed.shape)
    # print([node_name_map[each] for each in full_drug_set])
    m = best_embed[[node_name_map[each] for each in full_drug_set]]

    if D:
        optimal_list, opt_sum = optimal_list_model(D,C,Dk,E,I,N,t)
        # print(optimal_list, opt_sum)
        if opt_sum:
            results = reconciled_list_model(opt_sum,D,C,Dk,E,I,N,t,Dc,Dp,m)
            txt_results = [(result[0], result[1], result[2], 
                            [idx_to_drug[d] for d in result[3]],
                            [idx_to_drug[d] for d in result[4]]) for result in results] 
            return txt_results
        
    return []

if __name__ == '__main__':

    import pickle

    # %%
    df_full = pd.read_pickle('data/df_med.pkl')
    df_query = pd.read_excel('data/df_labeled_700.xlsx')
    # df_query = df_query.iloc[:100,:]   
    df_query['atcs'] = df_full.set_index('visID').loc[df_query.visID.values,'med'].values
    def code_trans(code):
        if len(code) == 7:
            return code[:5]+code[-1]
        else:
            return code
    df_query.icd9 = df_query.icd9.apply(lambda x: [code_trans(code) for code in x.replace("['",'').replace("']",'').replace("'",'').split(', ')])

    # %%

    uri = "bolt://localhost:7687"  # Adjust with Neo4j URI
    username = "neo4j"              # Replace with username
    password = "****"           # Replace with password

    neo4j_graph = py2neo.Graph(host='localhost', port=7687, user=username, password=password, name='combine2')

    # %%
    with open('data/node_name_map.pkl','rb') as f:
        node_name_map = pickle.load(f)
    import torch
    best_embed = torch.load('data/drug_emed_new.pt')
    df_query = df_query.iloc[:1,:]
    # print(df_query)
    df_query['reconciled_result'] = df_query.apply(lambda row: reconciled_list_generator(row, best_embed,neo4j_graph), axis = 1)

    df_query.to_excel('data/reconciled_list.xlsx')

    df_query.to_pickle('data/reconciled_list.pkl')
        