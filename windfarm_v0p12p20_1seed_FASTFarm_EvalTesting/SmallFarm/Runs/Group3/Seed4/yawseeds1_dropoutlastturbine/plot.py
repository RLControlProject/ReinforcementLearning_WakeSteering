import marimo

__generated_with = "0.15.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import plotly.graph_objects as go
    from pathlib import Path
    from scipy.signal import butter, filtfilt
    def butter_lowpass(cutoff, fs, order=4):
        nyq = 0.5 * fs  # Nyquist frequency
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a
    return Path, butter_lowpass, filtfilt, go, mo, pd


@app.cell
def _(Path, pd):
    mfile = Path('sac_output/trials/trial_5_10env_1_1/monitor.csv')
    dfm = pd.read_csv(mfile,skiprows=[0])
    return (dfm,)


@app.cell
def _(mo):
    # Low-pass filter design
    fs = 1
    cutoff = mo.ui.slider(.005,.05,0.001,label='cutoff')
    order = mo.ui.slider(1,10,label='order')
    return cutoff, fs, order


@app.cell
def _(butter_lowpass, cutoff, fs, order):
    b, a = butter_lowpass(cutoff.value, fs, order.value)
    return a, b


@app.cell
def _(cutoff):
    cutoff
    return


@app.cell
def _(order):
    order
    return


@app.cell
def _(a, b, dfm, filtfilt, go):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dfm.index,y=dfm['r'],opacity=0.4,name='Data'))
    fig.add_trace(go.Scatter(x=dfm.index,y = filtfilt(b, a, dfm['r']),name='Filtered'))
    fig.update_layout(
        title="Rewards",
        xaxis_title="Iteration",
        yaxis_title="Reward (%)"
    )
    fig.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
