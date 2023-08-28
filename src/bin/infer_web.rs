use axum::Server;
use tokio::runtime::Builder;

fn main() {
    let rt = Builder::new_current_thread().enable_all().build().unwrap();

    rt.block_on(async {
        if std::env::var_os("RUST_LOG").is_none() {
            std::env::set_var("RUST_LOG", "infer_web=info")
        }

        tracing_subscriber::fmt::init();

        let app = hello_tf::app().await;
        let addr = "0.0.0.0:3000".parse().unwrap();

        tracing::debug!("listening on {}", addr);

        Server::bind(&addr)
            .serve(app.into_make_service())
            .await
            .unwrap();
    })
}
