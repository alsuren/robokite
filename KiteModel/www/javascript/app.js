console.log("This is a kite simulator");
function liftCoefficient(alpha){
  // This is a simplified formula for lift coefficient
  // Fit with lift for an infinite elliptic wing for small angle
  // Maximum lift at 45°
  // No lift at 90°
  // Negative lift from 90°
  Cl = Math.PI*Math.sin (2*alpha);
  return Cl;
 }
function dragCoefficient(alpha){
  Cd0 = 0.01;
  Cd = Math.pow(Math.sin (alpha),2) + Cd0;
  return Cd;
}
var V = 10;

line_length     = 10;
wind_velocity   = 10;
kite_mass       = 1;  
kite_surface    = 6;  
rho_air         = 1;    // Air density
elevation0      = 0;
omega0          = 0;    // Angular rate
AoKdeg          = 50;   // Angle of Keying (calage)
sampleTime      = 0.001; // Sample time
g               = 9.81;
meter2pix = 20;

AoK = AoKdeg*Math.PI/180;
omega = 0;
elevation = 0;
y_base = 0;
z_base = 0;


// y is horizontal and positive in wind propagation direction
// z is vertical and positive up

// Base velocity relative to ground projected in ground axis
// Base is assumed to be static
v_base = 0;
w_base = 0;
document.getElementById("angleOfKeyRange").addEventListener("change", updateAngleOfKey);
document.getElementById("lineLengthRange").addEventListener("change", updateLineLength);
document.getElementById("windVelocityRange").addEventListener("change", updateWindVelocity);
document.getElementById("kiteMassRange").addEventListener("change", updateKiteMass);
document.getElementById("kiteSurfaceRange").addEventListener("change", updateKiteSurface);
setInterval(update, 1);
setInterval(updatePlot,100);
var d = new Date();
var told = d.getTime();

function plot(y_base, z_base, y_kite, z_kite, pitch){
  rotateKite(pitch);
  translateKite(y_kite, z_kite);
}
function updatePlot(){
  plot(y_base, z_base, y_kite, z_kite, pitch);
}

//function update(dt, AoK){
function update(){
  
  // Try to use real time
  d = new Date();
  t = d.getTime();
  dt = (t-told)/1000;
  told = t;
  console.log(dt/sampleTime)
  // Use constant sampleTime instead (to avoid Nan for unknown reason)
  dt = sampleTime// +0*dt;

  // Compute kite position
  y_kite = y_base + line_length * Math.cos(elevation);
  z_kite = z_base + line_length * Math.sin(elevation);
  
  pitch = AoK -elevation;
  //console.log(pitch);

  // Kite velocity relative to ground, projected in ground axis
  // Line length is assumed to be constant, and line is assumed straight.
  v_kite = v_base - omega*line_length*Math.cos(Math.PI/2 - elevation);
  w_kite = w_base + omega*line_length*Math.sin(Math.PI/2 - elevation);

  // Wind velocity: air velocity relative to ground, projected in ground axis
  // Assumed to be constant in time and space and horizontal
  v_wind = wind_velocity;
  w_wind = 0;

  // Wind relative velocity : air velocity relative to kite, projected in ground axis
  v_air_kite = v_wind-v_kite;
  w_air_kite = w_wind-w_kite;

  // Angle of attack of the kite, defined between kite chord and relative air velocity
  angle_air_kite = Math.atan2(w_air_kite, v_air_kite);
  AoA = angle_air_kite +pitch;
 
  // Dynamic pressure
  q = 1/2*rho_air *(v_air_kite*v_air_kite + w_air_kite*w_air_kite);

  // Lift and drag are in apparent wind frame
  lift   = q*kite_surface*liftCoefficient(AoA);
  drag   = q*kite_surface*dragCoefficient(AoA);
  
  // Rotate to ground frame
  Fz = lift* Math.cos(angle_air_kite) + drag*Math.sin(angle_air_kite);
  Fy = -lift* Math.sin(angle_air_kite) + drag*Math.cos(angle_air_kite);

  // Torque computed at base
  ML = +Fz * y_kite-kite_mass*g*y_kite;
  MD = -Fy * z_kite;

  // Angular acceleration
  omegap = 1/(kite_mass*line_length^2) * (ML + MD)- 0.0*omega;  //x*omega = amortissement
  //console.log(omegap);
  // Saturate to avoid instabilities
  omegap = Math.max(-60000, Math.min(omegap,60000));

  omega = omega + omegap * dt;
  
  // Saturate to avoid divergences 
  //console.log(omega);
  omega = Math.max(-60, Math.min(omega,60));
  elevation = elevation + omega * dt;
  y_base = y_base + v_base*dt;
  z_base = z_base + w_base*dt;
}
function rotateKite(r){
    kite = document.getElementById("kite");
        r_deg = r*180/Math.PI;
		kite.setAttribute('transform', 'rotate(' +r_deg +')');
		}
function translateKite(y, z){
  kite_frame = document.getElementById("local_frame");
  kite_line = document.getElementById("kite_line");
  kite_frame.setAttribute('transform', 'translate(' +y*meter2pix +','+ -z*meter2pix +')');
  kite_line.setAttribute('x2', y*meter2pix);
  kite_line.setAttribute('y2', -z*meter2pix);
}

function updateAngleOfKey(){
		//get elements
		var myRange = document.getElementById("angleOfKeyRange");
		var myOutput = document.getElementById("angleOfKey");
		//copy the value over
		myOutput.value = myRange.value;
    AoK = myOutput.value*Math.PI/180;
	}
function updateLineLength(){
		//get elements
		var myRange = document.getElementById("lineLengthRange");
		var myOutput = document.getElementById("lineLength");
		//copy the value over
		myOutput.value = myRange.value;
    line_length = myOutput.value;
	}
  function updateWindVelocity(){
		//get elements
		var myRange = document.getElementById("windVelocityRange");
		var myOutput = document.getElementById("windVelocity");
		//copy the value over
		myOutput.value = myRange.value;
    wind_velocity = myOutput.value;
	}
  function updateKiteMass(){
		//get elements
		var myRange = document.getElementById("kiteMassRange");
		var myOutput = document.getElementById("kiteMass");
		//copy the value over
		myOutput.value = myRange.value;
    kite_mass = myOutput.value;
	}
  function updateKiteSurface(){
		//get elements
		var myRange = document.getElementById("kiteSurfaceRange");
		var myOutput = document.getElementById("kiteSurface");
		//copy the value over
		myOutput.value = myRange.value;
    kite_surface = myOutput.value;
	}

