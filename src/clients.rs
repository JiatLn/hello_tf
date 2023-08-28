use crate::{infer::infer_client::InferClient, process::procss_client::ProcssClient};
use tonic::transport::Channel;

#[derive(Clone)]
pub struct Clients {
    pub infer_cli: InferClient<Channel>,
    pub process_cli: ProcssClient<Channel>,
}

impl Clients {
    pub async fn new() -> Self {
        let infer_cli = InferClient::connect("http://localhost:5000").await.unwrap();
        let process_cli = ProcssClient::connect("http://localhost:5002")
            .await
            .unwrap();

        Clients {
            infer_cli,
            process_cli,
        }
    }
}
