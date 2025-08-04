# Weather Prediction ML Add-on

Machine learning-based weather prediction for Home Assistant with 82.2% accuracy.

## Features

- 🤖 **Ensemble ML Model**: Combines XGBoost and Random Forest
- 📊 **82.2% Accuracy**: Validated on real weather data
- 🔄 **Automatic Updates**: Configurable prediction intervals
- 📡 **MQTT Autodiscovery**: Sensors appear automatically
- 📈 **Multiple Sensors**: Predictions, trends, confidence, and probabilities

## Installation

1. Add this repository to your Home Assistant Add-on Store
2. Install the "Weather Prediction ML" add-on
3. Configure your sensor entities
4. Start the add-on

## Configuration

- **mqtt_broker**: MQTT broker hostname (default: core-mosquitto)
- **mqtt_port**: MQTT broker port (default: 1883)
- **mqtt_username**: MQTT username (default: addons)
- **mqtt_password**: MQTT password
- **update_interval**: Prediction update interval in seconds (default: 3600)
- **temperature_sensor**: Entity ID of temperature sensor
- **humidity_sensor**: Entity ID of humidity sensor
- **pressure_sensor**: Entity ID of pressure sensor
- **log_level**: Logging level (DEBUG, INFO, WARNING, ERROR)

## Sensors Created

After starting the add-on, the following sensors will be automatically created:

- `sensor.weather_prediction` - Main prediction (increase/decrease/stable)
- `sensor.weather_trend` - Visual trend indicator (↑/↓/→)
- `sensor.prediction_confidence` - Confidence percentage
- `sensor.temperature_increase_probability` - Probability of temperature increase
- `sensor.temperature_decrease_probability` - Probability of temperature decrease
- `sensor.temperature_stable_probability` - Probability of stable temperature
- `sensor.last_prediction_update` - Timestamp of last update
- `sensor.model_accuracy` - Model accuracy (static 82.2%)

## Requirements

- Home Assistant OS or Supervised installation
- Mosquitto broker add-on (or compatible MQTT broker)
- Temperature, humidity, and pressure sensors with at least 48 hours of history

## Model Information

- **Type**: Ensemble (XGBoost 50.9% + Random Forest 49.1%)
- **Features**: 119 engineered features including temporal patterns, lags, and interactions
- **Training**: Trained on extensive historical weather data
- **Prediction**: 24-hour temperature trend forecast

## Support

For issues and feature requests, please open an issue on GitHub.