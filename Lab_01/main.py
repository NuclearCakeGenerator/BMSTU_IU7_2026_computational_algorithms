import matplotlib.pyplot as plt
import numpy as np
from utils import (
    load_current_data,
    load_heavy_particles_concentration,
    load_electrical_conductivity,
    load_heat_capacity,
    load_radiation_power,
    interpolate_2d,
    interpolate_1d,
)


class configuration:
    idle_pressure: float = 0.04  # mbar
    idle_temperature: float = 300  # K
    starting_time: float = 14e-6  # s
    ending_time: float = 450e-6  # s
    starting_temperature: float = 5400  # K
    tube_length: float = 0.1  # m
    tube_radius: float = 0.02  # m
    time_step: float = 1e-6  # s
    pressure_seek_range: tuple[float, float] = (0.3, 2.5)  # mbar


class wanted_data:
    temperature: list[tuple[float, float]] = []
    pressure: list[tuple[float, float]] = []
    conductivity: list[tuple[float, float]] = []
    volumetric_radiation: list[tuple[float, float]] = []
    # idle computation below
    resistance: list[tuple[float, float]] = []
    surface_radiation: list[tuple[float, float]] = []


class initial_data:
    current_t: list[tuple[float, float]] = load_current_data()
    concentration_T_P: list[tuple[float, float, float]] = (
        load_heavy_particles_concentration()
    )
    conductivity_T_P: list[tuple[float, float, float]] = load_electrical_conductivity()
    heat_capacity_T_P: list[tuple[float, float, float]] = load_heat_capacity()
    radiation_power_T_P: list[tuple[float, float, float]] = load_radiation_power()


def pressure_by_temperature(temperature: float) -> float:
    EQUALITY_ACCURACY = 1e-14
    pressure_min = configuration.pressure_seek_range[0]
    pressure_max = configuration.pressure_seek_range[1]
    while pressure_max - pressure_min > EQUALITY_ACCURACY:
        pressure_mid = (pressure_min + pressure_max) / 2
        concentration = interpolate_2d(
            initial_data.concentration_T_P, (temperature, pressure_mid)
        )

        if (
            concentration
            - 7.242e4 * configuration.idle_pressure / configuration.idle_temperature
            > 0
        ):
            pressure_max = pressure_mid
        else:
            pressure_min = pressure_mid

    return (pressure_min + pressure_max) / 2


def temperature_derivative(temperature: float, time: float) -> float:
    pressure = pressure_by_temperature(temperature)
    conductivity = interpolate_2d(
        initial_data.conductivity_T_P, (temperature, pressure)
    )
    radiation_power = interpolate_2d(
        initial_data.radiation_power_T_P, (temperature, pressure)
    )
    heat_capacity = interpolate_2d(
        initial_data.heat_capacity_T_P, (temperature, pressure)
    )
    current_density = interpolate_1d(initial_data.current_t, time) / (
        configuration.tube_radius**2 * np.pi
    )

    return (current_density**2 / conductivity - radiation_power) / heat_capacity


wanted_data.temperature.append(
    (configuration.starting_time, configuration.starting_temperature)
)
wanted_data.pressure.append(
    (
        configuration.starting_time,
        pressure_by_temperature(configuration.starting_temperature),
    )
)
wanted_data.conductivity.append(
    (
        configuration.starting_time,
        interpolate_2d(
            initial_data.conductivity_T_P,
            (configuration.starting_temperature, wanted_data.pressure[0][1]),
        ),
    )
)
wanted_data.volumetric_radiation.append(
    (
        configuration.starting_time,
        interpolate_2d(
            initial_data.radiation_power_T_P,
            (configuration.starting_temperature, wanted_data.pressure[0][1]),
        ),
    )
)

# main integration loop
last_time = configuration.starting_time
last_temperature = configuration.starting_temperature
while last_time <= configuration.ending_time:
    T_n_plus_1_2 = (
        last_temperature
        + temperature_derivative(last_temperature, last_time)
        * configuration.time_step
        / 2
    )
    P_n_plus_1_2 = pressure_by_temperature(T_n_plus_1_2)
    T_n_plus_1 = (
        last_temperature
        + temperature_derivative(T_n_plus_1_2, last_time + configuration.time_step / 2)
        * configuration.time_step
    )

    current_time = last_time + configuration.time_step
    current_temperature = T_n_plus_1
    current_pressure = pressure_by_temperature(current_temperature)
    current_conductivity = interpolate_2d(
        initial_data.conductivity_T_P, (current_temperature, current_pressure)
    )
    current_radiation = interpolate_2d(
        initial_data.radiation_power_T_P, (current_temperature, current_pressure)
    )

    wanted_data.temperature.append((current_time, current_temperature))
    wanted_data.pressure.append((current_time, current_pressure))
    wanted_data.conductivity.append((current_time, current_conductivity))
    wanted_data.volumetric_radiation.append((current_time, current_radiation))

    last_time = current_time
    last_temperature = current_temperature

wanted_data.resistance = [
    (
        t,
        configuration.tube_length
        / (conductivity * np.pi * configuration.tube_radius**2),
    )
    for t, conductivity in wanted_data.conductivity
]
wanted_data.surface_radiation = [
    (t, radiation * configuration.tube_radius / configuration.tube_length)
    for t, radiation in wanted_data.volumetric_radiation
]
