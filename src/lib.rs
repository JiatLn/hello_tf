mod clients;
mod handlers;
mod infer;
mod process;

use axum::{
    routing::{get, post},
    Router,
};
use clients::Clients;

pub async fn app() -> Router {
    let clients = Clients::new().await;

    Router::new()
        .route("/", get(handlers::root))
        .route("/classify_image", post(handlers::classify_image))
        .layer(tower_http::add_extension::AddExtensionLayer::new(clients))
        .layer(tower_http::trace::TraceLayer::new_for_http())
}
