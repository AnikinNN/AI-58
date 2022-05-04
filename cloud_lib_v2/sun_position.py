"""
base was taken from:
https://levelup.gitconnected.com/python-sun-position-for-solar-energy-and-research-7a4ead801777
"""
import math
import pandas as pd


def sunpos(when, location, refraction: bool = False):
    """
    calculate sun position depending on observer's time and location

    Args:
        when: tuple(year, month, day, hour, minute, second, timezone) or
              pd.Datetime UTC
        location: tuple(latitude, longitude)
        refraction: boolean, refraction correction (optional)

    Returns:
        tuple(azimuth, elevation)
    """
    # Extract the passed data
    if isinstance(when, tuple):
        year, month, day, hour, minute, second, timezone = when
    elif isinstance(when, pd.datetime):
        year = when.year
        month = when.month
        day = when.day
        hour = when.hour
        minute = when.minute
        second = when.second
        timezone = 0
    else:
        raise ValueError(f'when must be pd.datetime or tuple, got {type(when)}')
    # Convert latitude and longitude to radians
    lat, lon = map(math.radians, location)
    # Math typing shortcuts
    sin, cos, tan = math.sin, math.cos, math.tan
    asin, atan2 = math.asin, math.atan2
    # Decimal hour of the day at Greenwich
    greenwich_time = hour - timezone + minute / 60 + second / 3600
    # Days from J2000, accurate from 1901 to 2099
    day_num = (
            367 * year
            - 7 * (year + (month + 9) // 12) // 4
            + 275 * month // 9
            + day
            - 730531.5
            + greenwich_time / 24
    )
    # Mean longitude of the sun
    mean_long = day_num * 0.01720279239 + 4.894967873
    # Mean anomaly of the Sun
    mean_anom = day_num * 0.01720197034 + 6.240040768
    # Ecliptic longitude of the sun
    eclipse_long = (
            mean_long
            + 0.03342305518 * sin(mean_anom)
            + 0.0003490658504 * sin(2 * mean_anom)
    )
    # Obliquity of the ecliptic
    obliquity = 0.4090877234 - 0.000000006981317008 * day_num
    # Right ascension of the sun
    rasc = atan2(cos(obliquity) * sin(eclipse_long), cos(eclipse_long))
    # Declination of the sun
    decl = asin(sin(obliquity) * sin(eclipse_long))
    # Local sidereal time
    sidereal = 4.894961213 + 6.300388099 * day_num + lon
    # Hour angle of the sun
    hour_ang = sidereal - rasc
    # Local elevation of the sun
    elevation = asin(sin(decl) * sin(lat) + cos(decl) * cos(lat) * cos(hour_ang))
    # Local azimuth of the sun
    azimuth = atan2(
        -cos(decl) * cos(lat) * sin(hour_ang),
        sin(decl) - sin(lat) * sin(elevation),
    )
    # Convert azimuth and elevation to degrees
    azimuth = into_range(math.degrees(azimuth), 0, 360)
    elevation = into_range(math.degrees(elevation), -180, 180)
    # Refraction correction (optional)
    if refraction:
        targ = math.radians((elevation + (10.3 / (elevation + 5.11))))
        elevation += (1.02 / tan(targ)) / 60
    # Return azimuth and elevation in degrees
    return round(azimuth, 2), round(elevation, 2)


def into_range(x, range_min, range_max):
    shifted_x = x - range_min
    delta = range_max - range_min
    return (((shifted_x % delta) + delta) % delta) + range_min


if __name__ == "__main__":
    # Close Encounters latitude, longitude
    location = (40.602778, -104.741667)
    # Fourth of July, 2022 at 11:20 am MDT (-6 hours)
    when = (2022, 7, 4, 11, 20, 0, -6)
    # when = pd.to_datetime('2022/07/04 11:20:0')
    # Get the Sun's apparent location in the sky
    azimuth, elevation = sunpos(when, location, True)
    # Output the results
    print("\nWhen: ", when)
    print("Where: ", location)
    print("Azimuth: ", azimuth)
    print("Elevation: ", elevation)
# When:  (2022, 7, 4, 11, 20, 0, -6)
# Where:  (40.602778, -104.741667)
# Azimuth:  121.38
# Elevation:  61.91
