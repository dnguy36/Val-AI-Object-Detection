import pygetwindow as gw

# List all open windows
windows = gw.getAllTitles()
for window in windows:
    print(window)
