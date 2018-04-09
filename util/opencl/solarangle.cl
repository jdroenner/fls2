#define dEarthMeanRadius     6371.01	// In km
#define dAstronomicalUnit    149597890	// In km

int position(int x, int y, int x_size){
	return y*x_size + x;
}

double2 satelliteViewAngleToLatLon(double2 satelliteViewAngle, double sub_lon){
	double2 xy = radians(satelliteViewAngle);

	double2 sinxy = sin(xy);
	double2 cosxy = cos(xy);

	double cos2y = pown(cosxy.y, 2);
	double sin2y = pown(sinxy.y, 2);
	double cosxcosy = cosxy.x * cosxy.y;
	double cos2yconstsin2y = cos2y + 1.006803 * sin2y;

	double sd_part1 = pown(42164 * cosxcosy, 2);
	double sd_part2 = cos2yconstsin2y * 1737121856;
	double sd = sqrt(sd_part1 - sd_part2);
	double sn_part1 = 42164 * cosxcosy - sd;
	double sn_part2 = cos2yconstsin2y;
	double sn = sn_part1 / sn_part2;
	double s1 = 42164 - sn * cosxcosy;
	double s2 = sn * sinxy.x * cosxy.y;
	double s3 = -1.0 * sn * sinxy.y;
	double sxy = sqrt(pown(s1, 2) + pown(s2, 2));

	double2 latLonRad;
	double lonRad_part = s2 / s1;
	latLonRad.y = atan(lonRad_part) + radians(sub_lon);
	double latRad_part = 1.006804 * s3 / sxy;
	latLonRad.x = -atan(latRad_part);

	return degrees(latLonRad);
}

double2 solarAzimuthZenith(double dGreenwichMeanSiderealTime, double dRightAscension, double dDeclination, double2 dLatLon){

		double2 azimuthZenith;

		double dLocalMeanSiderealTime = radians(dGreenwichMeanSiderealTime*15 + dLatLon.y);;
		double dHourAngle = dLocalMeanSiderealTime - dRightAscension;
		double dLatitudeInRadians = radians(dLatLon.x);
		double dCos_Latitude = cos( dLatitudeInRadians );
		double dSin_Latitude = sin( dLatitudeInRadians );
		double dCos_HourAngle= cos( dHourAngle );

		azimuthZenith.y = (acos( dCos_Latitude*dCos_HourAngle*cos(dDeclination) + sin( dDeclination )*dSin_Latitude));


		double dY = -sin( dHourAngle );
		double dX = tan( dDeclination )*dCos_Latitude - dSin_Latitude*dCos_HourAngle;
		azimuthZenith.x = atan2( dY, dX );

		if ( azimuthZenith.x < 0.0 )
			azimuthZenith.x = azimuthZenith.x + M_PI*2;
		//udtSunCoordinates->dAzimuth = udtSunCoordinates->dAzimuth/rad;
		// Parallax Correction
		double dParallax=(dEarthMeanRadius/dAstronomicalUnit)*sin(azimuthZenith.y);
		azimuthZenith.y=(azimuthZenith.y + dParallax);///rad;

	return degrees(azimuthZenith);
}

__kernel void zenithKernel(__global float *out_data, const float dGreenwichMeanSiderealTime, const float dRightAscension, const float dDeclination, const float scale_x, const float scale_y, const float origin_x, const float origin_y, const float sub_sat) {
	int xsize = get_global_size(1);
	int ysize = get_global_size(0);
	int xgid = get_global_id(1);
	int ygid = get_global_id(0);
	int gid = position(xgid,ygid,xsize);
	double toViewAngleFac = 65536.0 / (-13642337.0 * (double)scale_x);
	
	double2 geosPosition;
	geosPosition.y = ((ygid-origin_y) * (double)scale_y);
	geosPosition.x = ((xgid-origin_x) * (double)scale_x);
	
	double2 satelliteViewAngle = geosPosition * toViewAngleFac * (double2)(-1,1);
	double2 latLonPosition = satelliteViewAngleToLatLon(satelliteViewAngle, ((double) sub_sat));
	double2 azimuthZenith = solarAzimuthZenith((double)dGreenwichMeanSiderealTime, (double)dRightAscension, (double)dDeclination, latLonPosition);
	
	out_data[gid] = azimuthZenith.y;
}