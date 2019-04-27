// Copyright (c) 2013 Nautilabs
// https://github.com/baptistelabat/robokite

// This file used SoftwareSerial example (Copyright (c) 2012 Dimension Engineering LLC)
// for Sabertooth http://www.dimensionengineering.com/software/SabertoothSimplifiedArduinoLibrary/html/index.html
// See license.txt in the Sabertooth arduino library for license details.
//
// The Sabertooth selector 1 3 5 6 have to be on on.

// Includes
#include <SoftwareSerial.h> 
#include <SabertoothSimplified.h> // From http://www.dimensionengineering.com/software/SabertoothArduinoLibraries.zip
#include <TinyGPS++.h>    // From https://github.com/mikalhart/TinyGPSPlus
#include <PID_v1.h>       // From https://github.com/br3ttb/Arduino-PID-Library/tree/master/PID_v1
#include <SPI.h> // Not actualy used but needed to compile
#include <MavlinkForArduino.h>

// Port definition
#define      TX_PIN 8 // Plug to Sabertooth RX
#define      RX_PIN 10 // Do not plug
#define RF_DATA_PIN 11
#define     LED_PIN 13 // Built-in led


// String management
String inputString = "";         // a string to hold incoming data
boolean stringComplete = false;  // whether the string is complete

// Sabertooth connection
// Note: NOT_A_PIN (0) was previously used for RX which is not used, but this was causing problems
SoftwareSerial SWSerial(RX_PIN, TX_PIN); // RX, TX. RX on no pin (unused), TX on pin 8 (to S1).
SabertoothSimplified ST(SWSerial); // Use SWSerial as the serial port.
int power1, power2;

// NMEA protocol for robust messages
TinyGPSPlus nmea;
// O stands for Opensource, R for Robokite
// The index is the place of the field in the NMEA message
TinyGPSCustom feedback_request      (nmea, "ORFBR", 1);  // Feedback request. The feedback are normalized between 0 and 1023
// The orders are normalized between -127 and 127
TinyGPSCustom pwm1     (nmea, "ORPW1", 1);  // Dimentionless voltage setpoint (Pulse Width Modulation) for Sabertooth output 1
TinyGPSCustom pwm2     (nmea, "ORPW2", 1);  // Dimentionless voltage setpoint (Pulse Width Modulation) for Sabertooth output 2
TinyGPSCustom setpos1  (nmea, "ORSP1", 1);  // Position setpoint for Sabertooth output 1
TinyGPSCustom setpos2  (nmea, "ORSP2", 1);  // Position setpoint for Sabertooth output 2
TinyGPSCustom speedlim1  (nmea, "ORSL1", 1);  // Speed limitation for Sabertooth output 1
TinyGPSCustom speedlim2  (nmea, "ORSL2", 1);  // Speed limitation for Sabertooth output 2
TinyGPSCustom poslim1  (nmea, "ORPL1", 1);  // Position limitation for Sabertooth output 1
TinyGPSCustom poslim2  (nmea, "ORPL2", 1);  // Position limitation for Sabertooth output 2
TinyGPSCustom kpm1 (nmea, "ORKP1", 1); // Proportional coefficient multiplicator
//TinyGPSCustom kim1 (nmea, "ORKI1", 1); // Integral coefficient multiplicator
TinyGPSCustom kdm1 (nmea, "ORKD1", 1); // Derivative coefficient multiplicator
//TinyGPSCustom kpm2 (nmea, "ORKP2", 1); // Proportional coefficient multiplicator
//TinyGPSCustom kim2 (nmea, "ORKI2", 1); // Integral coefficient multiplicator
//TinyGPSCustom kdm2 (nmea, "ORKD2", 1); // Derivative coefficient multiplicator
boolean isFeedbackRequested = false;

uint8_t data[4];  // 2 element array of unsigned 8-bit type, holding Joystick readings



// PID for robust control
// Define Variables we'll be connecting to
double Setpoint1, Input1, Output1;
double Setpoint2, Input2, Output2;
double Input3, Input4, Input5, Input6;
float posSat1 = 1;
float posSat2 = 1;
float speedSat1 = 1;
float speedSat2 = 1;
// Specify the links and initial tuning parameters (Kp, Ki, Kd)
float Kp1 = 1;
float Ki1 = 0.00;
float Kd1 = 0.001;
float Kp2 = 1;
float Ki2 = 0.00;
float Kd2 = 0.001;
PID myPID1(&Input1, &Output1, &Setpoint1, Kp1, Ki1, Kd1, DIRECT);
PID myPID2(&Input2, &Output2, &Setpoint2, Kp2, Ki2, Kd2, DIRECT);


#define ORDER_RATE_ms 100
long last_order_ms = 0;
long last_simu_ms = 0;
int dt_ms = 0;

boolean SIL = false; // Use Software in the Loop simulation

/* HX711 part */
#include "HX711.h"

#define HX711_DOUT  12 // or DAT
#define HX711_CLK   11
#define HX711_GND   13

#define OFFSET -797000 // From raw value when no load
#define SCALE  37.0  // From calibration
HX711 scale;
long raw_value;
void setup()
{
  setupHX711();
  pinMode(LED_PIN, OUTPUT); 
  
  // Initialize software serial communication with Sabertooth 
  SWSerial.begin(9600);
  
  // Initialize serial communications with computer
  Serial.begin(115200);
  
  // Reserve bytes for the inputString:
  inputString.reserve(200);
  
  // Initialize input and setpoint
  Input1 = 0;
  Input2 = 0;
  Input3 = 0;

  Setpoint1 = 0;
  Setpoint2 = 0;
  
  // Turn the PID on
  myPID1.SetMode(MANUAL);
  myPID2.SetMode(MANUAL);
  // Saturate the output to the maximum range (values are normalized between -1 and 1)
  myPID1.SetOutputLimits(-1, 1);
  myPID2.SetOutputLimits(-1, 1);
  
  for (int i=0;i<4;i++)
  {
    data[i] = 127;
  }
  
  last_simu_ms = millis();
}

void setupHX711() {
  // Ground connection
  scale.begin(HX711_DOUT, HX711_CLK);
  pinMode(HX711_GND, OUTPUT);
  Serial.print("HX711_DOUT -> D");
  Serial.println(HX711_DOUT);
  Serial.print("HX711_CLK -> D");
  Serial.println(HX711_CLK);
  Serial.print("HX711_GND -> D");
  Serial.println(HX711_GND);
  Serial.println("HX711_VCC -> 3V3");

   // No calibration, read the raw value
  scale.set_offset(0);
  scale.set_scale(1);
}

/*
  SerialEvent occurs whenever a new data comes in the
 hardware serial RX.  This routine is run between each
 time loop() runs, so using delay inside loop can delay
 response.  Multiple bytes of data may be available.
 */
void serialEvent()
{
  while (Serial.available()) {
    // Get the new byte:
    char inChar = (char)Serial.read(); 
    // Add it to the inputString:
    inputString += inChar;
    // If the incoming character is a newline, set a flag
    // so the main loop can do something about it:
    if (inChar == '\n') {
      stringComplete = true;
    } 
  }
}

void loop()
{
   processSerialInput();
   processInput();
   if (false==SIL)
   {
     computeFeedback();
   }
   computeOrder();
   sendOrder();
   if (SIL)
   {
     simulate();
   }
   sendFeedback();
   delay(10);
}
void processInput()
{
  data[0] = scale.get_units();
  Input1 = (data[0]-OFFSET)/SCALE;


  Input4 = analogRead(A0)/1023.;
  Input5 = analogRead(A1)/1023.;
  Input6 = analogRead(A2)/1023.;
}

void processSerialInput()
{
  // Print the string when a newline arrives:
  if (stringComplete)
  {   
    // Serial.println(inputString);
    char ctab[inputString.length()+1];
    inputString.toCharArray(ctab, sizeof(ctab));
    char *gpsStream = ctab;

    while (*gpsStream)
    {
      if (nmea.encode(*gpsStream++))
      {
        if (pwm1.isUpdated())
        { 
          pwm1.value();
          myPID1.SetMode(MANUAL);
        }
        if (pwm2.isUpdated())
        {
          pwm2.value();
          myPID2.SetMode(MANUAL);
        }
        if (setpos1.isUpdated())
        { 
          setpos1.value();
          myPID1.SetMode(AUTOMATIC);
        }
        if (setpos2.isUpdated())
        {
          setpos2.value();
          myPID2.SetMode(AUTOMATIC);
        }
        if (feedback_request.isUpdated())
        {
          feedback_request.value();
          // $ORFBR,0*57
          isFeedbackRequested = true;
          digitalWrite(LED_PIN, HIGH);
        }
        if (kpm1.isUpdated()||kdm1.isUpdated())//||kpm2.isUpdated()||kim2.isUpdated()||kdm2.isUpdated())
        {
          kpm1.value();
          kdm1.value();
          computePIDTuning();
        }
        if (speedlim1.isUpdated()||speedlim2.isUpdated()||poslim1.isUpdated()||poslim2.isUpdated())
        {
          updateSaturation();
          speedlim1.value();
          speedlim2.value();
          poslim1.value();
          poslim2.value();
        }  
      }
    }
    // Clear the string:
    inputString = "";
    stringComplete = false;
  }
}

void computeFeedback()
{
  
  // Feedbacks are normalized between -1 and 1
  
  // Potentiometer angle
  float rawAngle_deg = (data[0]-127)/255.0;
  
}

void sendFeedback()
{
  /*mavlink_message_t msg; 
  uint8_t bufout[MAVLINK_MAX_PACKET_LEN];
  uint16_t len;
  uint64_t time_us = micros();
  int group_mlx = 0;
  uint8_t system_id = 100;
  uint8_t target_system = system_id;
  uint8_t component_id = 1;
  uint8_t target_component = 2;// Sabertooth
  */
  // All the feedback values are normalized in the range 0-1023 (10 bits resolution)
  if (isFeedbackRequested)
  {
    
    float orders[8];
    orders[0] = float(power1/127.);
    orders[1] = float(power2/127.);
    //mavlink_msg_set_actuator_control_target_pack(system_id, component_id, &msg, time_us, group_mlx, target_system, target_component, orders);
    //len = mavlink_msg_to_send_buffer(bufout, &msg);
    //Serial.write(bufout, len);
    
    Serial.print((power1+127)*4);
    Serial.print(", ");
    Serial.print((power2+127)*4);
    Serial.print(", ");
    Serial.print((Input1+1)/2.*1023);
    Serial.print(", ");
    Serial.print((Input2+1)/2.*1023);
    Serial.print(", ");
    Serial.print((Input3+1)/2.*1023);
    Serial.print(", ");
    Serial.print((Input4+1)/2.*1023);
    Serial.print(", ");
    Serial.print((Input5+1)/2.*1023);
    Serial.print(", ");
    Serial.print((Input6+1)/2.*1023);
    Serial.println("");
    isFeedbackRequested = false;
  }
}

void computeOrder()
{
  if (myPID1.GetMode() == MANUAL)
  {
    power1 = atoi(pwm1.value())*speedSat1;
  }
  if (myPID2.GetMode() == MANUAL)
  {
    power2 = atoi(pwm2.value())*speedSat2;
  }
  if (myPID1.GetMode() == AUTOMATIC)
  {
    Setpoint1 = atoi(setpos1.value())/127.*posSat1*1.3;//Factor to go a bit further and be sure to reach saturation (even with static error)
    myPID1.Compute();
    power1 = Output1*127;
  }
  if (myPID2.GetMode() == AUTOMATIC)
  {
    Setpoint2 = atoi(setpos2.value())/127.*posSat2*1.3;
    myPID2.Compute();
    power2 = Output2*127;
  }
  if (Input1>posSat1)
  {
    power1 = min(power1,0);
  }
  if (Input1<-posSat1)
  {
    power1 = max(power1,0);
  }
}

void sendOrder()
{
  if (fabs(millis()-last_order_ms)>ORDER_RATE_ms)
  {
    last_order_ms = millis();
  // Order in the range -127/127
    //SWSerial.println(127);
      //Saturation based on posSat
   ST.motor(1, power1);
   ST.motor(2, power2);
  }
}
void computePIDTuning()
{
   myPID1.SetTunings(fabs(atof(kpm1.value())/127.*Kp1), 0, fabs(atof(kdm1.value())/127.*Kd1));
 
 //myPID1.SetTunings(fabs(atof(kpm1.value())/127.*Kp1), fabs(atof(kim1.value())/127.*Ki1), fabs(atof(kdm1.value())/127.*Kd1));
 //myPID2.SetTunings(fabs(atof(kpm2.value())/127.*Kp2), fabs(atof(kim2.value())/127.*Ki2), fabs(atof(kdm2.value())/127.*Kd2));
 myPID1.SetControllerDirection(sgn(atoi(kpm1.value())));
 //myPID2.SetControllerDirection(sgn(atoi(kpm2.value())));
 //Serial.println(atof(kpm1.value())/127.*Kp1);
}
void updateSaturation()
{
  if (speedlim1.isUpdated())
  {
     myPID1.SetOutputLimits(-atoi(speedlim1.value())/127., atoi(speedlim1.value())/127.);
     speedSat1 = atoi(speedlim1.value())/127.;
  }
  if (speedlim2.isUpdated())
  {
    myPID2.SetOutputLimits(-atoi(speedlim2.value())/127., atoi(speedlim2.value())/127.);
  }
  if (poslim1.isUpdated())
  {
    posSat1 = atoi(poslim1.value())/127.;
  }
  //posSat2 = atoi(poslim2.value())/127.;
}
static inline int8_t sgn(int val) {
  if (val < 0) return -1;
  if (val==0) return 0;
  return 1;
}
 
void simulate()
{ dt_ms = millis()-last_simu_ms;
  Input1=Input1+power1/127.*dt_ms/1000.;
  Input2=Input2+power2/127.*dt_ms/1000.;
  last_simu_ms= millis();
}
