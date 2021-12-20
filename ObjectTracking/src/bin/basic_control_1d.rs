// import time
use std::thread::sleep;
use std::time::Duration;
use std::io::Write;

// from SimpleCV import *
use opencv::core::{MatExpr, Point, Scalar, Size, ToInputArray, no_array, KeyPoint, Vector};
use opencv::imgproc::MORPH_CLOSE;
use opencv::prelude::*;
use opencv::{highgui, imgproc, videoio, ximgproc, features2d};

// import serial
trait MatExt {
    fn invert(&self) -> Result<MatExpr, anyhow::Error>;
    fn threshold(&self, thresh: f64) -> Result<Mat, anyhow::Error>;
    fn find_blobs(&self, max_size: f64) -> Result<Vector<KeyPoint>, anyhow::Error>;
}

impl MatExt for Mat {
    // 1 - self
    fn invert(&self) -> Result<MatExpr, anyhow::Error> {
        opencv::core::sub_scalar_mat(Scalar::from(1.0), self).map_err(Into::into)
    }
    // from SimpleCV, which is BSD 3-Clause
    fn threshold(&self, thresh: f64) -> Result<Mat, anyhow::Error> {
        // FIXME: gray = self._getGrayscaleBitmap()
        let mut result = Mat::default();
        imgproc::threshold(self, &mut result, thresh, 1.0, imgproc::THRESH_BINARY)?;
        Ok(result)
    }
    // from SimpleCV, which is BSD 3-Clause
    fn find_blobs(&self, max_size: f64) -> Result<Vector<KeyPoint>, anyhow::Error> {
        // FIXME:
        // if (maxsize == 0):
        //     maxsize = self.width * self.height
        // #create a single channel image, thresholded to parameters
        //
        // blobmaker = BlobMaker()
        let mut blobmaker = features2d::SimpleBlobDetector::create(features2d::SimpleBlobDetector_Params::default()?)?;
        let mut blobs = Default::default();
        blobmaker.detect(self, &mut blobs, &no_array())?;
        // blobs = blobmaker.extractFromBinary(
        //     self.binarize(
        //         threshval, 255, threshblocksize, threshconstant
        //     ).invert(),
        //     self, minsize = minsize, maxsize = maxsize,appx_level=appx_level,
        // )
        //
        // if not len(blobs):
        //     return None
        //
        // return FeatureSet(blobs).sortArea()

        Ok(blobs)
    }
}

trait ToInputArrayExt {
    fn hue_distance(&self, color: u8) -> Result<Mat, anyhow::Error>;
    fn morph_close(&self) -> Result<Mat, anyhow::Error>;
}

impl<T: ToInputArray> ToInputArrayExt for T {
    // from SimpleCV, which is BSD 3-Clause
    fn hue_distance(&self, color: u8) -> Result<Mat, anyhow::Error> {
        let mut hsv = Mat::default();
        imgproc::cvt_color(self, &mut hsv, imgproc::COLOR_BGR2HSV, 0)?;
        todo!()
    }
    // from SimpleCV, which is BSD 3-Clause
    fn morph_close(&self) -> Result<Mat, anyhow::Error> {
        let mut ret = Mat::default();
        let temp = Mat::default();
        // kern = cv.CreateStructuringElementEx(3, 3, 1, 1, cv.CV_SHAPE_RECT);
        let kern = ximgproc::get_structuring_element(imgproc::MORPH_RECT, Size::new(3, 3))?;
        // cv.MorphologyEx(self.getBitmap(), ret, temp, kern, cv.MORPH_CLOSE, 1)
        ximgproc::morphology_ex(self, &mut ret, MORPH_CLOSE, &kern, true, Point::new(-1, -1))?;
        Ok(ret)
    }
}

fn main() -> Result<(), anyhow::Error> {
    // cam = JpegStreamCamera('http://192.168.1.6:8080/videofeed')
    let mut cam = videoio::VideoCapture::default()?;
    // disp=Display()
    let window = "video capture";
    highgui::named_window(window, 1)?;

    // """This script was used for the demonstration of doing control with visual feedback
    //    A android mobile phone was used with ipcam application to stream the video
    //    A green fresbee was attached to a line rolled over the axis of the motor which was controlled"""

    // ser = serial.Serial('/dev/ttyACM2', 9600)
    let mut ser = std::io::stdout();

    let alpha = 0.8;

    sleep(Duration::from_secs(1));
    let mut previous_z = 200f32;

    loop {
        // img = cam.getImage()
        let mut img = Mat::default();
        cam.read(&mut img)?;

        // myLayer = DrawingLayer((img.width,img.height))
        // disk_img = img.hueDistance(color=Color.GREEN).invert().morphClose().morphClose().threshold(200)
        let disk_img = img
            .hue_distance(1u8)?
            .invert()?
            .morph_close()?
            .morph_close()?
            .threshold(200.0 / 255.0)?;
        let disk = disk_img.find_blobs(2000.0)?;
        if !disk.is_empty() {
            // disk[0].drawMinRect(layer=myLayer, color=Color.RED)
            // disk_img.addDrawingLayer(myLayer)
            let position = disk.get(0)?.pt;
            dbg!(&position);
            let z = alpha*position.y+(1.0-alpha)*previous_z;
            write!(ser, "{}", (z-200.0)*0.03)?;
            previous_z=z;
        }
        // disk_img.save(disp)
        sleep(Duration::from_millis(10));

    }

}
