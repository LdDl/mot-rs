use kalman_rust::kalman;
use std::fmt;

#[derive(Debug)]
pub enum TrackerError {
    KalmanError(kalman::Kalman2DError),
    KalmanBBoxError(kalman::KalmanBBoxError),
    NoObject(NoObjectInTracker),
    BadSize(String),
}

impl From<kalman::Kalman2DError> for TrackerError {
    fn from(e: kalman::Kalman2DError) -> Self {
        TrackerError::KalmanError(e)
    }
}

impl From<kalman::KalmanBBoxError> for TrackerError {
    fn from(e: kalman::KalmanBBoxError) -> Self {
        TrackerError::KalmanBBoxError(e)
    }
}

impl From<NoObjectInTracker> for TrackerError {
    fn from(e: NoObjectInTracker) -> Self {
        TrackerError::NoObject(e)
    }
}

impl From<String> for TrackerError {
    fn from(e: String) -> Self {
        TrackerError::BadSize(e)
    }
}

#[derive(Debug)]
pub struct NoObjectInTracker {
    pub txt: String,
}
impl fmt::Display for NoObjectInTracker {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "NoObjectInTracker: {}", self.txt)
    }
}
