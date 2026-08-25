# 02 – Event cleaning and sensor identity

**Question:** how are heterogeneous provider rows converted into valid comparable events?

The maintained core command is:

```powershell
python scripts\01_build_event_master.py --config config\demo.yaml
```

Replace the configuration with `config/scientific.yaml` for authorised standardised inputs. Output: `<output_root>/01_cleaning/clean_events.csv`.

The cleaning contract normalises company/permit strings, parses UTC timestamps, recomputes duration, rejects missing/invalid chronology and de-duplicates identical company–permit–timestamp events. `sensor_uid` is `company::permit_number`.

The historical `scripts/prepare_sensor_master.py` is a later all-sensor assembly step that consumes completed fit/bootstrap/location products; despite its historical filename it is not the maintained event-cleaning entry point.

Validation tests:

```powershell
pytest tests\test_duration_calculation.py 02_data_cleaning\tests\test_event_contracts.py -q
```
