use axum::extract::Multipart;
use axum::response::IntoResponse;
use axum::{self, Extension, Json};
use axum::{routing::get, routing::post, Router};
use hello_tf::{
    infer_client::InferClient, procss_client::ProcssClient, InferRequest, InferResponse,
    PostProcessRequest, PostProcessResponse, PreProcessRequest, PreProcessResponse,
};
use tokio::runtime::Builder;
use tonic::transport::Channel;
use tower_http::add_extension::AddExtensionLayer;

#[derive(Clone)]
struct Clients {
    infer_cli: InferClient<Channel>,
    process_cli: ProcssClient<Channel>,
}

fn main() {
    let rt = Builder::new_current_thread().enable_all().build().unwrap();

    rt.block_on(async {
        if std::env::var_os("RUST_LOG").is_none() {
            std::env::set_var("RUST_LOG", "infer_web=info")
        }

        tracing_subscriber::fmt::init();

        let clients = Clients {
            infer_cli: InferClient::connect("http://localhost:5000").await.unwrap(),
            process_cli: ProcssClient::connect("http://localhost:5002")
                .await
                .unwrap(),
        };

        let app = Router::new()
            .route("/", get(root))
            .route("/", post(classify_image))
            .layer(AddExtensionLayer::new(clients))
            .layer(tower_http::trace::TraceLayer::new_for_http());

        let addr = "0.0.0.0:3000".parse().unwrap();

        tracing::debug!("listening on {}", addr);

        axum::Server::bind(&addr)
            .serve(app.into_make_service())
            .await
            .unwrap();
    })
}

// basic handler that responds with a static string
async fn root() -> &'static str {
    "Hello, World!"
}

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

async fn classify_image(
    Extension(clients): Extension<Clients>,
    mut multipart: Multipart,
) -> impl IntoResponse {
    let Clients {
        mut infer_cli,
        mut process_cli,
    } = clients;
    let mut results = vec![];
    while let Some(field) = multipart.next_field().await.unwrap() {
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
        results.push(ClassifyResult { image, preds })
    }
    Json(results)
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
