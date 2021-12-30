// import time
use std::thread::sleep;
use std::time::Duration;

// from SimpleCV import *
use color_eyre::eyre;
use opencv::core::{
    abs_matexpr, min, no_array, sub_scalar_mat, KeyPoint, Point, Scalar, Size, ToInputArray,
    Vector, _InputArrayTraitConst, mix_channels, BorderTypes, CV_32FC3, CV_8UC3,
};
use opencv::features2d::{self, SimpleBlobDetector, SimpleBlobDetector_Params};
use opencv::imgproc::MORPH_CLOSE;
use opencv::prelude::*;
use opencv::{highgui, imgproc, videoio};
use tracing::field::valuable;
use tracing::{info, warn};

// import serial
trait MatExt {
    fn to_float(&self) -> Result<Mat, eyre::Error>;
    fn to_8bit(&self) -> Result<Mat, eyre::Error>;
    fn flatten_hue(&self) -> Result<Mat, eyre::Error>;
    fn hue_distance(&self) -> Result<Mat, eyre::Error>;
    fn invert(&self) -> Result<Mat, eyre::Error>;
    fn threshold(&self, thresh: f64) -> Result<Mat, eyre::Error>;
    fn find_blobs(&self, max_size: f32) -> Result<Vector<KeyPoint>, eyre::Error>;
}

impl MatExt for Mat {
    fn to_float(&self) -> Result<Mat, eyre::Error> {
        let mut ret = Mat::default();
        self.convert_to(&mut ret, CV_32FC3, 1. / 255., 0.)?;
        Ok(ret)
    }
    fn to_8bit(&self) -> Result<Mat, eyre::Error> {
        let mut ret = Mat::default();
        self.convert_to(&mut ret, CV_8UC3, 255., 0.)?;
        Ok(ret)
    }
    fn flatten_hue(&self) -> Result<Mat, eyre::Error> {
        let mut hsv_channels = Mat::default();
        imgproc::cvt_color(self, &mut hsv_channels, imgproc::COLOR_BGR2HSV, 1)?;

        let mut hue_channel = Mat::new_rows_cols_with_default(
            hsv_channels.rows(),
            hsv_channels.cols(),
            CV_32FC3,
            // set Saturation and Value to 0.5 so we can see things;
            Scalar::new(0., 0.5, 0.5, 1.),
        )?;
        mix_channels(&hsv_channels, &mut hue_channel, &[0, 0])?;

        let mut ret = Mat::default();
        imgproc::cvt_color(&hue_channel, &mut ret, imgproc::COLOR_HSV2BGR, 0)?;
        Ok(ret)
    }
    // from SimpleCV, which is BSD 3-Clause
    fn hue_distance(&self /* TODO: color: ???*/) -> Result<Mat, eyre::Error> {
        assert!(!self.empty());

        let mut hsv_channels = Mat::default();
        imgproc::cvt_color(self, &mut hsv_channels, imgproc::COLOR_BGR2HSV, 1)?;

        let mut hue_channel = Mat::new_rows_cols_with_default(
            hsv_channels.rows(),
            hsv_channels.cols(),
            CV_32FC3,
            // set Saturation and Value to 0.5 so we can see things;
            Scalar::new(0., 0.5, 0.5, 1.0),
        )?;
        mix_channels(&hsv_channels, &mut hue_channel, &[0, 0])?;
        // turquoise, because I happen to have turquoise things near me.
        let color_hue = 180.;
        let color_hue_wrapped = 180. + 360.;
        let mut distance = Mat::default();
        min(
            &abs_matexpr(&sub_scalar_mat(
                Scalar::new(color_hue, 0.5, 0.5, 1.0),
                &hue_channel,
            )?)?,
            &abs_matexpr(&sub_scalar_mat(
                Scalar::new(color_hue_wrapped, 0.5, 0.5, 1.0),
                &hue_channel,
            )?)?,
            &mut distance,
        )?;
        // FIXME: I feel like SimpleCV has some better treatment here, like clipping to 1.
        let mut tmp = Mat::default();
        distance.convert_to(&mut tmp, CV_32FC3, 1. / 30., 0.)?;
        let mut ret = Mat::default();
        imgproc::cvt_color(&tmp, &mut ret, imgproc::COLOR_BGR2GRAY, 1)?;

        Ok(ret)
    }
    // 1 - self
    fn invert(&self) -> Result<Mat, eyre::Error> {
        assert!(!self.empty());
        let expr = opencv::core::sub_scalar_mat(Scalar::new(1.0, 1.0, 1.0, 1.0), self)?;
        let mat = expr.input_array()?.get_mat(-1)?;
        Ok(mat)
    }
    // from SimpleCV, which is BSD 3-Clause
    fn threshold(&self, thresh: f64) -> Result<Mat, eyre::Error> {
        assert!(!self.empty());
        // FIXME: gray = self._getGrayscaleBitmap()
        let mut result = Mat::default();
        imgproc::threshold(self, &mut result, thresh, 1.0, imgproc::THRESH_BINARY)?;
        Ok(result)
    }
    // from SimpleCV, which is BSD 3-Clause
    fn find_blobs(&self, max_area: f32) -> Result<Vector<KeyPoint>, eyre::Error> {
        assert!(!self.empty());
        // FIXME:
        // if (maxsize == 0):
        //     maxsize = self.width * self.height
        // #create a single channel image, thresholded to parameters
        //
        // blobmaker = BlobMaker()
        let mut blobmaker = SimpleBlobDetector::create(SimpleBlobDetector_Params {
            // FIXME: this is a bit of a hack.
            min_area: max_area / 20.0,
            max_area,
            ..SimpleBlobDetector_Params::default()?
        })?;
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
        let mut blobs = blobs.to_vec();
        blobs.sort_by_key(|b| float_ord::FloatOrd(b.size));
        let blobs = Vector::from_iter(blobs.into_iter());
        Ok(blobs)
    }
}

trait ToInputArrayExt {
    fn morph_close(&self) -> Result<Mat, eyre::Error>;
}

impl<T: ToInputArray> ToInputArrayExt for T {
    // from SimpleCV, which is BSD 3-Clause
    fn morph_close(&self) -> Result<Mat, eyre::Error> {
        let mut ret = Mat::default();
        // kern = cv.CreateStructuringElementEx(3, 3, 1, 1, cv.CV_SHAPE_RECT);
        let kern = imgproc::get_structuring_element(
            imgproc::MORPH_RECT,
            Size::new(3, 3),
            Point::new(-1, -1),
        )?;
        // cv.MorphologyEx(self.getBitmap(), ret, temp, kern, cv.MORPH_CLOSE, 1)
        imgproc::morphology_ex(
            self,
            &mut ret,
            MORPH_CLOSE,
            &kern,
            Point::new(-1, -1),
            1,
            BorderTypes::BORDER_CONSTANT as i32,
            imgproc::morphology_default_border_value()?,
        )?;
        Ok(ret)
    }
}

struct BGRImage(Mat);

impl valuable::Valuable for BGRImage {
    fn as_value(&self) -> valuable::Value<'_> {
        todo!()
    }

    fn visit(&self, visit: &mut dyn valuable::Visit) {
        todo!()
    }
}

fn main() -> eyre::Result<()> {
    color_eyre::install()?;
    tracing_subscriber::fmt::init();

    // cam = JpegStreamCamera('http://192.168.1.6:8080/videofeed')
    let mut cam = videoio::VideoCapture::new(0, videoio::CAP_ANY)?;
    assert!(videoio::VideoCapture::is_opened(&cam)?);
    // disp=Display()
    let window = "video capture";
    highgui::named_window(window, 1)?;

    // """This script was used for the demonstration of doing control with visual feedback
    //    A android mobile phone was used with ipcam application to stream the video
    //    A green fresbee was attached to a line rolled over the axis of the motor which was controlled"""

    // ser = serial.Serial('/dev/ttyACM2', 9600)
    // let mut ser = std::io::stdout();

    let alpha = 0.8;

    sleep(Duration::from_secs(1));
    let mut previous_z = 200f32;

    loop {
        // img = cam.getImage()
        let mut img = Mat::default();
        cam.read(&mut img)?;

        // myLayer = DrawingLayer((img.width,img.height))
        // disk_img = img.hueDistance(color=Color.GREEN).invert().morphClose().morphClose().threshold(200)
        // let disk_img = img.to_float()?.flatten_hue()?;
        let mut disk_img = img
            .to_float()?
            .hue_distance()?
            .invert()?
            .morph_close()?
            .morph_close()?
            .threshold(200.0 / 255.0)?
            .to_8bit()?;

        let disk = disk_img.find_blobs(2000.0)?;
        if !disk.is_empty() {
            // disk[0].drawMinRect(layer=myLayer, color=Color.RED)
            // disk_img.addDrawingLayer(myLayer)
            features2d::draw_keypoints(
                &disk_img.clone(),
                &disk,
                &mut disk_img,
                Scalar::all(-1f64),
                features2d::DrawMatchesFlags::DEFAULT,
            )?;
            let position = disk.get(0)?.pt;
            dbg!(&position);
            let z = alpha * position.y + (1.0 - alpha) * previous_z;
            // ser.write(str((z-200)*0.03))
            println!("{}", (z - 200.0) * 0.03);
            previous_z = z;
        }
        warn!("hi there");
        warn!(disk_img = valuable(BGRImage(disk_img.clone())));
        highgui::imshow(window, &disk_img)?;
        if highgui::wait_key(10)? > 0 || true {
            break;
        }
    }
    Ok(())
}
