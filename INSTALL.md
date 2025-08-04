# Installation Instructions for Weather Prediction ML Add-on

## Prerequisites

1. **Home Assistant OS or Supervised** - Add-ons only work with these installation types
2. **Mosquitto broker add-on** - Already installed (confirmed in your setup)
3. **Sensor History** - At least 48 hours of temperature, humidity, and pressure data

## Installation Steps

### Step 1: Copy Add-on to Home Assistant

```bash
# From your local machine
scp -r weather-prediction-ml-addon hassio@192.168.31.114:/addons/
```

### Step 2: Add Local Add-on Repository

1. In Home Assistant, go to **Settings** → **Add-ons**
2. Click **Add-on Store**
3. Click the **⋮** (three dots) menu → **Repositories**
4. Add: `/addons/weather-prediction-ml-addon`
5. Click **Add**

### Step 3: Install the Add-on

1. Refresh the Add-on Store (pull down to refresh)
2. Look for "Weather Prediction ML" in the Local Add-ons section
3. Click on it and then click **Install**
4. Wait for installation to complete (this may take a few minutes)

### Step 4: Configure the Add-on

1. After installation, go to the **Configuration** tab
2. Set your sensor entity IDs:
   ```yaml
   temperature_sensor: sensor.sensor_varanda_temperature
   humidity_sensor: sensor.sensor_varanda_humidity
   pressure_sensor: sensor.sensor_varanda_pressure
   ```
3. Leave other settings as default (they're already configured for your Mosquitto setup)
4. Click **Save**

### Step 5: Start the Add-on

1. Go to the **Info** tab
2. Click **Start**
3. Enable **Start on boot** and **Watchdog**
4. Check the **Log** tab to ensure it's running correctly

### Step 6: Verify Sensors

The add-on will automatically create sensors via MQTT discovery. Check for:

1. Go to **Settings** → **Devices & Services** → **MQTT**
2. You should see a new device: "Weather Prediction ML"
3. Click on it to see all created sensors

## Alternative: Docker Build (Advanced)

If you want to build the Docker image locally:

```bash
cd weather-prediction-ml-addon
docker build --build-arg BUILD_FROM="ghcr.io/home-assistant/amd64-base:3.19" -t weather-prediction-ml .
```

## Troubleshooting

### Add-on Not Showing
- Ensure the folder is in `/addons/` not `/config/addons/`
- Refresh the Add-on Store
- Check folder permissions

### Start Fails
- Check the Log tab for errors
- Verify sensor entity IDs exist
- Ensure MQTT broker is running

### No Sensors Created
- Wait 1-2 minutes after starting
- Check MQTT integration for new devices
- Verify in Developer Tools → States

### Insufficient Data Error
- Ensure your sensors have 48+ hours of history
- Check sensor data in History tab
- Wait for more data to accumulate

## Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| mqtt_broker | core-mosquitto | MQTT broker hostname |
| mqtt_port | 1883 | MQTT broker port |
| mqtt_username | addons | MQTT username |
| mqtt_password | (empty) | MQTT password |
| update_interval | 3600 | Prediction interval (seconds) |
| temperature_sensor | (required) | Temperature entity ID |
| humidity_sensor | (required) | Humidity entity ID |
| pressure_sensor | (required) | Pressure entity ID |
| log_level | INFO | Logging level |

## Next Steps

1. Add sensors to your dashboard
2. Create automations based on predictions
3. Monitor prediction accuracy over time