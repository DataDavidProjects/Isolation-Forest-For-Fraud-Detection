from ibm_watson_machine_learning import APIClient

def ibm_model(
        fields,
        values,
        api_key = '6kp6CmCFTD4TRkFgW_QJLUXWLokaPwAnj2ADo7s673dR',
        location = "eu-gb",
        deployment_uid = '9b3a85e0-76cf-4f7b-82c4-272464237280',
        WML_SPACE_ID="d9bc5247-7acd-4601-85fd-910e2c49f299",
              ):

    wml_credentials = {
        "apikey": api_key,
        "url": 'https://' + location + '.ml.cloud.ibm.com'
    }
    client = APIClient(wml_credentials)
    client.set.default_space(WML_SPACE_ID)
    scoring_payload = {"input_data": [{"fields": fields, "values": values}]}
    #scoring_payload
    predictions = client.deployments.score(deployment_uid, scoring_payload)
    return predictions




username = 'davide_lupis@yahoo.it'
api_key = 'dvbUwNS8pnLHavMvFzgxC67R6bm9AvLSYdm8h3S6CZDU'
location = "eu-gb"
url = 'https://' + location + '.ml.cloud.ibm.com'
wml_credentials = {
    "username": username,
    "apikey": api_key,
    "url": url,
    "instance_id": 'openshift',
    "version": '4.0'
}

client = APIClient(wml_credentials)

space_id = 'd9bc5247-7acd-4601-85fd-910e2c49f299'

client.set.default_space(space_id)


sofware_spec_uid = client.software_specifications.get_id_by_name("runtime-22.1-py3.9")

metadata = {
            client.repository.ModelMetaNames.NAME: 'Scikit model',
            client.repository.ModelMetaNames.TYPE: 'scikit-learn_1.0',
            client.repository.ModelMetaNames.SOFTWARE_SPEC_UID: sofware_spec_uid
}

published_model = client.repository.store_model(
    model=model,
    meta_props=metadata)


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

