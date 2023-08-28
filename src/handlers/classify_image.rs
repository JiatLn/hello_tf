use crate::{
    clients::Clients,
    infer::{infer_client::InferClient, InferRequest, InferResponse},
    process::{
        procss_client::ProcssClient, PostProcessRequest, PostProcessResponse, PreProcessRequest,
        PreProcessResponse,
    },
};
use axum::{extract::Multipart, response::IntoResponse, Extension, Json};
use tonic::transport::Channel;

#[derive(serde::Serialize)]
struct Pred {
    name: String,
    prob: f32,
}

#[derive(serde::Serialize)]
struct ClassifyResult {
    image: String,
    preds: Vec<Pred>,
}

pub async fn classify_image(
    Extension(clients): Extension<Clients>,
    mut multipart: Multipart,
) -> impl IntoResponse {
    let Clients {
        mut infer_cli,
        mut process_cli,
    } = clients;
    if let Some(field) = multipart.next_field().await.unwrap() {
        let image = field.file_name().unwrap().to_string();
        let data = field.bytes().await.unwrap();

        let PreProcessResponse { data, shape } = pre_process(&mut process_cli, &data).await;
        let InferResponse { data, shape } = infer(&mut infer_cli, &data, &shape).await;
        let PostProcessResponse { preds } = post_process(&mut process_cli, &data, &shape).await;
        let preds = preds
            .into_iter()
            .map(|p| Pred {
                name: p.name,
                prob: p.prob,
            })
            .collect();
        println!("classify_image called");
        Json(ClassifyResult { image, preds })
    } else {
        Json(ClassifyResult {
            image: "None".to_string(),
            preds: vec![],
        })
    }
}

async fn pre_process(cli: &mut ProcssClient<Channel>, data: &[u8]) -> PreProcessResponse {
    let request = PreProcessRequest {
        image: data.to_vec(),
    };
    cli.pre_process(request).await.unwrap().into_inner()
}

async fn infer(cli: &mut InferClient<Channel>, data: &[f32], shape: &[u64]) -> InferResponse {
    let request = InferRequest {
        data: data.to_vec(),
        shape: shape.to_vec(),
    };
    cli.infer(request).await.unwrap().into_inner()
}

async fn post_process(
    cli: &mut ProcssClient<Channel>,
    data: &[f32],
    shape: &[u64],
) -> PostProcessResponse {
    let request = PostProcessRequest {
        data: data.to_vec(),
        shape: shape.to_vec(),
    };
    cli.post_process(request).await.unwrap().into_inner()
}
