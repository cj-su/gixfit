"""Scripted end-to-end run of the desktop app (opens the window, presses the buttons, closes it).

python tests/test_app_gui.py [out_dir]          (~10 min: identify screens the whole CIF library)
"""
import os, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main():
    import tkinter as tk
    from tkinter import messagebox
    from gixfit import app as A
    from gixfit.engine import Settings

    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "runs", "app_test")
    os.makedirs(out, exist_ok=True)
    msgs = []
    for name in ("showerror", "showinfo"):          # never block on a dialog
        setattr(messagebox, name, lambda *a, _n=name, **k: msgs.append((_n, a)))
    root = tk.Tk()
    app = A.App(root)

    def wait(label):
        t0 = time.time()
        root.update()
        while app.busy:
            root.update(); time.sleep(0.05)
        root.update()
        print(f"{label}: {time.time() - t0:.0f} s | status: {app.v_status.get()}", flush=True)

    s, img, variants = Settings.from_toml(os.path.join(ROOT, "examples", "npb_R1x1_th010.toml"), variant=1)
    app.settings = s
    app._fill_fields(s)
    app.v_image.set(img)
    app._run_find(); wait("Find spots")
    assert app.session is not None and len(app.session.peaks) > 50, msgs
    app.fig.savefig(os.path.join(out, "01_spots.png"), dpi=110)

    npb = os.path.join(ROOT, "library", "NPB_published_cell_DFT_atoms.cif")
    moo3 = os.path.join(ROOT, "library", "MoO3_alpha_Pbnm_Kihlborg1963_COD9014282.cif")
    app.v_chance.set("Quick (6 surrogates)")
    app.v_cif.set(npb); app.v_contact.set("1 0 1")
    app._run_check(); wait("Check one CIF (NPB, contact 1 0 1)")
    assert app.evals and app.evals[0].best is not None, msgs
    app._show_selected(); root.update()
    print("   ", app.evals[0].summary_line())
    print("    spot-error rows:", len(app.tv_err.get_children("")))
    app.fig.savefig(os.path.join(out, "02_check_npb.png"), dpi=110)

    for p in (npb, moo3):
        app.lb_cifs.insert("end", p)
    app._run_compare(); wait("Compare CIFs (NPB vs MoO3)")
    assert len(app.evals) == 2, msgs
    for e in app.evals:
        print("   ", e.summary_line())
    app.fig.savefig(os.path.join(out, "03_compare.png"), dpi=110)

    app.v_top.set("2")
    app._run_identify(); wait("Identify (library)")
    assert app.evals, msgs
    for e in app.evals[:5]:
        print("   ", e.summary_line())
    app.fig.savefig(os.path.join(out, "04_identify.png"), dpi=110)

    # export (bypass the folder dialog)
    from gixfit.engine import export
    export(app.session, app.evals, app.flags, os.path.join(out, "report"))
    print("report files:", sorted(os.listdir(os.path.join(out, "report")))[:8], "...")
    print("dialogs that would have popped up:", msgs or "none")
    root.destroy()
    print("APP END-TO-END OK")


if __name__ == "__main__":
    main()
