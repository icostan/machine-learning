import pandas as pd
import featuretools as ft

C_PATH='./input/'

def load_train_data():
    print('Loading CSV data...')
    applications_df = pd.read_csv(C_PATH + 'application_train.csv')
    previous_df = pd.read_csv(C_PATH + 'previous_application.csv')
    # bureau_df = pd.read_csv(C_PATH + 'bureau.csv')

    print("Creating entityset...")
    es = ft.EntitySet(id="home-credit")

    print("Loading applications entity...")
    es = es.entity_from_dataframe(entity_id="applications", dataframe=applications_df, index="SK_ID_CURR")
    print("Loading previous entity...")
    es = es.entity_from_dataframe(entity_id="previous", dataframe=previous_df, index="SK_ID_PREV")
    # print("Loading bureau data...")
    # es = es.entity_from_dataframe(entity_id="bureau", dataframe=bureau_df, index="SK_ID_BUREAU")

    print("Adding relationships...")
    applications_previous = ft.Relationship(es["applications"]["SK_ID_CURR"], es["previous"]["SK_ID_CURR"])
    es = es.add_relationship(applications_previous)
    # applications_bureau = ft.Relationship(es["applications"]["SK_ID_CURR"], es["bureau"]["SK_ID_CURR"])
    # es = es.add_relationship(applications_bureau)

    # return es

    print("Generating DFS...")
    feature_matrix, feature_defs = ft.dfs(entityset=es, target_entity="applications", verbose=True)
    fm_encoded, defs_encoded = ft.encode_features(feature_matrix, feature_defs)
    return fm_encoded, defs_encoded
