// Compared to the original version, this version fix the issue for unmatched entities

CALL apoc.export.json.query(
  "MATCH (d:Drug)
   OPTIONAL MATCH (d)-[:HAS_CODE*..3]-(a:ATC)
   OPTIONAL MATCH (d)-[:HAS_CODE*..3]-(c:CUI_Drug)
   WITH d, 
        collect(DISTINCT CASE WHEN a IS NOT NULL THEN a.code ELSE NULL END) AS atcs,
        collect(DISTINCT CASE WHEN c IS NOT NULL THEN c.cui ELSE NULL END) AS cuis
   RETURN d.drugbank_id AS drugbank_id, 
          d.name AS name, 
          d.description AS description, 
          d.indication AS indication, 
          atcs, 
          cuis
   UNION
   // ATC 与 CUI_Drug 的直接关系，无 Drug 关联
   MATCH (a:ATC)-[:HAS_CODE*..3]-(c:CUI_Drug)
   WHERE NOT EXISTS { MATCH (a)-[:HAS_CODE*..3]-(:Drug)}
   WITH a.code AS atc_code, 
        a.name AS atc_name,
        c
   RETURN 'UNMATCHED_ATC_' + atc_code AS drugbank_id, 
          atc_name AS name, 
          NULL AS description, 
          NULL AS indication, 
          [atc_code] AS atcs, 
          collect(DISTINCT CASE WHEN c IS NOT NULL THEN c.cui ELSE NULL END) AS cuis
   UNION
   // 孤立的 CUI_Drug，无关联 Drug 或 ATC
   MATCH (c:CUI_Drug)
   WHERE NOT EXISTS { MATCH (c)-[:HAS_CODE*..3]-(:Drug)} AND NOT EXISTS { MATCH (c)-[:HAS_CODE*..3]-(:ATC)}
   RETURN 'UNMATCHED_CUI_' + c.cui AS drugbank_id, 
          c.name AS name, 
          NULL AS description, 
          NULL AS indication, 
          [] AS atcs, 
          [c.cui] AS cuis",
  "DrugProfile2.json",
  {stream: true}
)


#########

CALL apoc.export.json.query(
     "
     // Diseases with their matched codes
     MATCH (d:CUI_Disease)
     OPTIONAL MATCH (d)-[:HAS_CODE*]-(i:ICD9CM)
     OPTIONAL MATCH (d)-[:HAS_CODE*]-(c:Disease)
     WITH d,
          collect(DISTINCT CASE WHEN i IS NOT NULL THEN i.code ELSE NULL END) AS icd9cms,
          collect(DISTINCT CASE WHEN c IS NOT NULL THEN c.cui ELSE NULL END) AS mondos
     RETURN d.cui AS id, d.name AS name, icd9cms, mondos

     UNION

     // Unmatched ICD9CM nodes (no link to CUI_Disease)
     MATCH (i:ICD9CM)
     WHERE NOT EXISTS { MATCH (i)-[:HAS_CODE*]-(:CUI_Disease) }
     OPTIONAL MATCH (i)-[:HAS_CODE*]-(c:Disease)
     RETURN 'UNMATCHED_' + i.code AS id, 
               i.name AS name, 
               [i.code] AS icd9cms,
               collect(DISTINCT CASE WHEN c IS NOT NULL THEN c.code ELSE NULL END) AS mondos
     UNION

     // Unmatched Disease nodes (no link to CUI_Disease)
     MATCH (c:Disease)
     WHERE NOT EXISTS { MATCH (c)-[:HAS_CODE*]-(:CUI_Disease) } 
     AND NOT EXISTS { MATCH (c)-[:HAS_CODE*]-(:ICD9CM) }
     RETURN 'UNMATCHED_' + c.code AS id, 
               c.name AS name, 
               [] AS icd9cms,
               [c.code] AS mondos
     ",

     "disease_profiles2.json",
     {stream: true}
)


##############
CREATE CONSTRAINT IF NOT EXISTS FOR (n: DrugProfile) REQUIRE (n.drugbank_id) IS UNIQUE;
CREATE INDEX IF NOT EXISTS FOR (n: DrugProfile) ON (n.name);
CREATE INDEX IF NOT EXISTS FOR (n:DrugProfile) ON (n.cuis);
CREATE INDEX IF NOT EXISTS FOR (n:DrugProfile) ON (n.atcs);

CALL apoc.load.json("file:///DrugProfile2.json") YIELD value
MERGE (profile:DrugProfile {drugbank_id: value.drugbank_id})
SET profile.name = value.name,
    profile.description = value.description,
    profile.indication = value.indication,
    profile.atcs = value.atcs,
    profile.cuis = value.cuis;

CREATE CONSTRAINT IF NOT EXISTS FOR (n: DiseaseProfile) REQUIRE (n.id) IS UNIQUE;
CREATE INDEX IF NOT EXISTS FOR (n: DiseaseProfile) ON (n.name);
CREATE INDEX IF NOT EXISTS FOR (n:DiseaseProfile) ON (n.icd9cms);
CREATE INDEX IF NOT EXISTS FOR (n:DiseaseProfile) ON (n.mondos);

CALL apoc.load.json("file:///disease_profiles2.json") YIELD value
MERGE (profile:DiseaseProfile {id: value.id})
SET profile.name = value.name,
    profile.icd9cms = value.icd9cms,
    profile.mondos = value.mondos;

:param file_path_root => 'http://localhost:11001/project-be34219e-694c-4f9f-b9ad-308deb82820a/';
:param file_3 => 'ddi_major_moderate.csv'; //ddi_selected_drugbank.csv
LOAD CSV WITH HEADERS FROM ($file_path_root + $file_3) AS row
WITH row
MATCH (source:DrugProfile { drugbank_id: row.drugbank_id })
MATCH (target:DrugProfile { drugbank_id: row.inter_id })
MERGE (source)-[i:INTERACTS]->(target)
SET i.description = row.inter_description;



:param file_11 => 'treated.csv';
LOAD CSV WITH HEADERS FROM ($file_path_root + $file_11) AS row
WITH row
MATCH (source:DrugProfile)
WHERE row.CUI1 IN source.cuis
MATCH (target:DiseaseProfile)
WHERE target.id = row.CUI2
MERGE (source)-[i:TREATS]->(target);

:param file_2 => 'reposdb.csv';
LOAD CSV WITH HEADERS FROM ($file_path_root + $file_2) AS row
WITH row
MATCH (source:DrugProfile { drugbank_id: row.drugbank_id })
MATCH (target:DiseaseProfile)
where target.id = row.ind_id
MERGE (source)-[i:TREATS]->(target);

:param file_12 => 'prime_kg_indication.csv';
LOAD CSV WITH HEADERS FROM ($file_path_root + $file_12) AS row
WITH row
MATCH (source:DrugProfile {drugbank_id: row.drug})
MATCH (target:DiseaseProfile)
WHERE  row.disease IN target.mondos 
MERGE (source)-[i:TREATS]->(target);

:param file_13 => 'drug_indications_05122020.tsv';
LOAD CSV WITH HEADERS FROM ($file_path_root + $file_13) AS row FIELDTERMINATOR '\t'
WITH row
MATCH (source:DrugProfile), (target:DiseaseProfile)
WHERE LOWER(source.name) = LOWER(row.DRUG_NAME)
AND target.id = row.UMLS_CUI
MERGE (source)-[i:TREATS]->(target);


:param file_9 => 'contraind.csv';
LOAD CSV WITH HEADERS FROM ($file_path_root + $file_9) AS row
WITH row
MATCH (source:DrugProfile)
WHERE row.CUI2 IN source.cuis 
MATCH (target:DiseaseProfile)
WHERE target.id = row.CUI1
MERGE (source)-[i:HAS_CONTRAIND]->(target);


:param file_5 => 'disease_parents.csv';
LOAD CSV WITH HEADERS FROM ($file_path_root + $file_5) AS row
WITH row 
MATCH (source:DiseaseProfile)
where row.id IN source.mondos
MATCH (target:DiseaseProfile)
where row.parents IN target.mondos 
AND source<>target
MERGE (source)-[r: HAS_PARENTCODE]->(target);


:param file_0 => 'ATC_final.csv';
LOAD CSV WITH HEADERS FROM ($file_path_root + $file_0) AS row
WITH row 
MATCH (source:DrugProfile)
where row.code IN  source.atcs
MATCH (target:DrugProfile)
WHERE row.parent_code IN target.atcs
AND source<>target
MERGE (source)-[r: HAS_PARENTCODE]->(target);


:param file_1 => 'icd9cm_full.csv';
LOAD CSV WITH HEADERS FROM ($file_path_root + $file_1) AS row
WITH row 
MATCH (source:DiseaseProfile)
where row.id IN source.icd9cms
MATCH (target:DiseaseProfile)
WHERE row.parents IN target.icd9cms
AND source<>target
MERGE (source)-[r: HAS_PARENTCODE]->(target);
