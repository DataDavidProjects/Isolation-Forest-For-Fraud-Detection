from src.scoring.credentials_api import *
from ibm_watson_machine_learning import APIClient

def store_model(model,client,
                space_id=space_id,
                model_name = "IsolationForest",
                model_type= "scikit-learn_1.1",
                sofware_spec_uid = None):
    client.set.default_space(space_id)
    if sofware_spec_uid is None:
        sofware_spec_uid = client.software_specifications.get_id_by_name("runtime-22.2-py3.10")
        print("Using standard runtime software spec")
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


def create_custom_env(space_id):
    client.set.default_space(space_id)
    pe_metadata = {
         client.package_extensions.ConfigurationMetaNames.NAME:
             'custom',
         # optional:
         # wml_client.software_specifications.ConfigurationMetaNames.DESCRIPTION:
         client.package_extensions.ConfigurationMetaNames.TYPE:
             'conda_yml'
    }
    pe_asset_details = client.package_extensions.store(
     meta_props=pe_metadata,
     file_path='custom.yaml'
    )

    pe_asset_id = client.package_extensions.get_id(pe_asset_details)
    pe_asset_id
    # Get the id of the base software specification
    base_id = client.software_specifications.get_id_by_name('runtime-22.2-py3.10')

    # create the metadata for software specs
    ss_metadata = {
     client.software_specifications.ConfigurationMetaNames.NAME:
         'custom python 3.10',
     client.software_specifications.ConfigurationMetaNames.DESCRIPTION:
         'custom libraries', # optional
     client.software_specifications.ConfigurationMetaNames.BASE_SOFTWARE_SPECIFICATION:
         {'guid': base_id},
     client.software_specifications.ConfigurationMetaNames.PACKAGE_EXTENSIONS:
         [{'guid': pe_asset_id}]
    }

    # store the software spec
    ss_asset_details = client.software_specifications.store(meta_props=ss_metadata)

    # get the id of the new asset
    sofware_spec_uid = client.software_specifications.get_id(ss_asset_details)
    return sofware_spec_uid


def erase_custom_env(n=10,env = 'custom python 3.10'):
    for _ in range(n):
        
        c = client.software_specifications.get_id_by_name(env)
        print(c)
        client.software_specifications.delete(c)