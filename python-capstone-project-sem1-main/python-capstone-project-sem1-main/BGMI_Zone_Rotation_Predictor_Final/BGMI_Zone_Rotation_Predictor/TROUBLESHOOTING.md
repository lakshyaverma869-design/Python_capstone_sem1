# Troubleshooting

## Python or package installation

Run setup.bat and check that 64-bit Python 3.11-3.14 is installed. First-time setup needs access to PyPI to install requirements.txt.

## Strategy generation is unavailable

The final ZIP does not include the external historical CSV because its license could not be established. If you have permission to use the data, save the file as data\historical_zone_data.csv with the columns Match_ID, Phase, White_X, White_Y, White_R, then restart with run.bat. Without enough valid same-phase matches, the application displays a clear message and does not fabricate a zone.

## Map interaction

Use the original map image. Drag the safe-zone circle, pull the edge handle to resize, click or drag to place the team marker, shift-drag to pan, and scroll to zoom. If the map component does not render, rerun setup.bat and ensure assets\map_component\index.html is present.
