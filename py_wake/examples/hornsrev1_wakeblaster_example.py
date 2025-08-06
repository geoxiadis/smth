from py_wake.examples.data.hornsrev1 import wt_x, wt_y, HornsrevV80,\
    Hornsrev1Site
from py_wake import WakeBlaster, WakeBlasterConfig
import matplotlib.pyplot as plt

def main():
    if __name__ == '__main__':
        import matplotlib.pyplot as plt
        wakeblasterconfig = WakeBlasterConfig()
        wakeblasterconfig.from_file("settings.json")

        wt = HornsrevV80()
        site = Hornsrev1Site()
        site.plot_wd_distribution(n_wd=12,ws_bins=[0,5,10,15,20,25])
        plt.show()
        wt.plot(wt_x, wt_y)
        wf_model = WakeBlaster(site, wt, wakeblasterconfig=wakeblasterconfig)
        sim_res = wf_model(wt_x, wt_y)
        aep = sim_res.aep()
        plt.title('AEP: %.1fGWh' % aep.sum())
        plt.show()
        
        print("Calculate Flow Map")
        flow_map = sim_res.flow_map(wd=120, ws=9.8)
        flow_map.plot_wake_map(plot_colorbar=False)
        flow_map.plot_windturbines()
        plt.title('WakeBlaster, AEP: %.2f GWh' % aep)
        plt.show()

main()
