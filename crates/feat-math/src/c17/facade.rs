// Safe typed facade over the model implementation in this module.
pub(crate) struct C17Model {
    model: Model,
    pub(crate) width: usize,
}
impl C17Model {
    pub(crate) fn fit(x: &[Vec<f64>], y: &[f64], width: usize) -> Self {
        Self {
            model: Model::fit(x, y),
            width,
        }
    }
    pub(crate) fn predict(&self, x: &[Vec<f64>]) -> Vec<f64> {
        self.model.predict(x)
    }
    pub(crate) fn score(&self, x: &[Vec<f64>], y: &[f64]) -> f64 {
        self.model.score(x, y)
    }
}
