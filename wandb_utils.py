import wandb

def log_df_as_table(df, table_name):
    if wandb.run is None:
        print('Initialize a W&B run')
        return

    # Log the df as Tables
    wandb.log({f"{table_name}": df})


def log_csv_as_artifact(path_to_csv, artifact_name, artifact_type='data'):
    if wandb.run is None:
        print('Initialize a W&B run')
        return
    
    # Log the df as Artifacts        
    wandb.log_artifact(path_to_csv, name=artifact_name, type=artifact_type) 
