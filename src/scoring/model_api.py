from src.scoring.credentials_api import *
from ibm_watson_machine_learning import APIClient

def store_model(model,client,
                space_id=space_id,
                model_name = "IsolationForest",
                model_type= 'scikit-learn_1.0',
                sofware_spec_uid = "runtime-22.1-py3.9"):
    client.set.default_space(space_id)
    sofware_spec_uid = client.software_specifications.get_id_by_name(sofware_spec_uid)
    metadata = {
        client.repository.ModelMetaNames.NAME: model_name,
        client.repository.ModelMetaNames.TYPE: model_type,
        client.repository.ModelMetaNames.SOFTWARE_SPEC_UID: sofware_spec_uid
    }
    published_model = client.repository.store_model(
        model=model,
        meta_props=metadata)
    return published_model


def deploy_model(name,published_model,client):
    metadata = {
        client.deployments.ConfigurationMetaNames.NAME: name,
        client.deployments.ConfigurationMetaNames.ONLINE: {}
    }

    published_model_uid = client.repository.get_model_id(published_model)
    created_deployment = client.deployments.create(published_model_uid,
                                                   meta_props=metadata)
    deployment_uid = client.deployments.get_id(created_deployment)

    return deployment_uid


def score_payload(fields,values,client,deployment_uid,space_id):
    client.set.default_space(space_id)
    scoring_payload = {"input_data": [{"fields": fields, "values": values}]}
    #scoring_payload
    predictions = client.deployments.score(deployment_uid, scoring_payload)
    return predictions


