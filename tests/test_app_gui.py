"""Scripted end-to-end run of the desktop app (opens the window, presses the buttons, closes it).

python tests/test_app_gui.py [out_dir]          (portable synthetic tutorial; diagnostics off)
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
    root.withdraw()
    app = A.App(root)

    def wait(label):
        t0 = time.time()
        root.update()
        while app.busy:
            root.update(); time.sleep(0.05)
        root.update()
        print(f"{label}: {time.time() - t0:.0f} s | status: {app.v_status.get()}", flush=True)

    from gixfit.demo import create_demo
    img, candidate, s = create_demo(out)
    app.settings = s
    app._fill_fields(s)
    app.v_image.set(img)
    app._run_find(); wait("Find spots")
    assert app.session is not None and len(app.session.peaks) > 20, msgs
    app.fig.savefig(os.path.join(out, "01_spots.png"), dpi=110)

    # Visible geometry changes must block fitting and must not mutate the existing session.
    old_x = app.session.geom.x0
    app.vars["x0"].set(str(old_x + 2))
    assert app._apply_options() is None
    assert app.session.geom.x0 == old_x and app.session.s.x0 == old_x
    app.vars["x0"].set(str(old_x))
    app.v_image.set(img + ".other")
    assert app._apply_options() is None
    app.v_image.set(img)
    assert app._apply_options() is not None
    msgs.clear()  # expected stale-input notices
    npb = candidate
    from gixfit.crystal import read_cif, write_cif, Cell
    model = read_cif(candidate)
    p = model.cell.array(); p[:3] *= 1.15
    moo3 = os.path.join(os.path.dirname(candidate), "wrong_candidate.cif")
    write_cif(model.with_cell(Cell.from_array(p)), moo3, "Wrong tutorial cell")
    app.v_chance.set("Off (fast fit)")
    app.v_cif.set(npb); app.v_contact.set("1 0 0")
    app._run_check(); wait("Check tutorial CIF (contact 1 0 0)")
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

    app.lb_lib.delete(0, "end")
    app.lb_lib.insert("end", os.path.dirname(candidate))
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
    assert not msgs, msgs
    root.destroy()
    print("APP END-TO-END OK")


if __name__ == "__main__":
    main()
