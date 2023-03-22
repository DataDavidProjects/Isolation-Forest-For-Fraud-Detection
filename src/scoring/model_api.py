from ibm_watson_machine_learning import APIClient

client = APIClient(wml_credentials)

space_id = 'PASTE YOUR SPACE ID HERE'

client.set.default_space(space_id)


sofware_spec_uid = client.software_specifications.get_id_by_name("runtime-22.1-py3.9")

metadata = {
            client.repository.ModelMetaNames.NAME: 'Scikit model',
            client.repository.ModelMetaNames.TYPE: 'scikit-learn_1.0',
            client.repository.ModelMetaNames.SOFTWARE_SPEC_UID: sofware_spec_uid
}

published_model = client.repository.store_model(
    model=model,
    meta_props=metadata,
    training_data=train_data,
    training_target=train_labels)


metadata = {
    client.deployments.ConfigurationMetaNames.NAME: "Deployment of scikit model",
    client.deployments.ConfigurationMetaNames.ONLINE: {}
}

published_model_uid = client.repository.get_model_id(published_model)
model_details = client.repository.get_details(published_model_uid)

created_deployment = client.deployments.create(published_model_uid, meta_props=metadata)

deployment_uid = client.deployments.get_id(created_deployment)

scoring_endpoint = client.deployments.get_scoring_href(created_deployment)
print(scoring_endpoint)


scoring_payload = {"input_data": [{"values": [ ]}]}

predictions = client.deployments.score(deployment_uid, scoring_payload)

