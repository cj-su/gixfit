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

    # click = add a result to the map, double-click = remove it (selection managed by the app, not by Tk)
    from types import SimpleNamespace
    rows = app.tv_res.get_children("")
    def at(iid, col):
        app.tv_res.see(iid); root.update()
        x, y, w, h = app.tv_res.bbox(iid, col)
        return x + 5, y + h // 2
    assert app._sel == [rows[0]], app._sel                      # first result shown after a run
    x, y = at(rows[1], "#2"); app.tv_res.event_generate("<Button-1>", x=x, y=y); root.update()
    assert app._sel == [rows[0], rows[1]] and set(app.tv_res.selection()) == set(rows[:2]), app._sel
    assert len(A.as_list(app._shown_ev)) == 2                    # both fits drawn on the map
    app._dismiss_result(SimpleNamespace(x=x, y=y)); root.update()      # = <Double-Button-1> on row 2
    assert app._sel == [rows[0]], app._sel
    x, y = at(rows[1], "#5"); app.tv_res.event_generate("<Button-1>", x=x, y=y); root.update()
    app._run_joint(); wait("Fit selected together")
    assert app.joints and app._sel == ["J0"] and app.tv_phase.get_children(""), (app._sel, msgs)
    print("   ", app.joints[0].verdict)
    app.fig.savefig(os.path.join(out, "03b_joint.png"), dpi=110)

    # Export figure starts from the zoom shown on screen
    app.ax.set_xlim(0.3, 1.2); app.ax.set_ylim(0.1, 0.9); root.update()
    app._export_figure(); root.update()
    o = app.fig_dialog.current()
    assert o.qxy_lim == (0.3, 1.2) and o.qz_lim == (0.1, 0.9), (o.qxy_lim, o.qz_lim)
    app.fig_dialog.destroy(); app._draw(app._shown_ev)

    # Export report dialog: choose results and files
    app._select([rows[0]])
    win = app._export(); win = app.export_dialog
    assert win.v_which.get() == "selected"
    win.v_parts["replay"].set(False); win.v_parts["cifs"].set(False); win.v_parts["settings"].set(False)
    rep2 = os.path.join(out, "report_choices")
    win.v_dir.set(rep2)
    assert win.go(); wait("Export (selected row, chosen files)")
    files = sorted(os.listdir(rep2))
    assert "gixfit_results.xlsx" in files and not any(f.endswith(".cif") for f in files) \
        and "run_manifest.json" not in files and "inputs" not in files, files
    assert sum(f.endswith("_map.png") for f in files) == 1, files      # only the selected result
    lib = os.path.join(out, "cif_library"); os.makedirs(lib, exist_ok=True)
    open(os.path.join(lib, "x.cif"), "w").close()
    asked = []
    mb_yesno = messagebox.askyesno
    messagebox.askyesno = lambda *a, **k: asked.append(a) or False
    win = app._export(); win = app.export_dialog; win.v_parts["cifs"].set(True); win.v_dir.set(lib)
    assert win.go() is False and asked and sorted(os.listdir(lib)) == ["x.cif"]    # warned, nothing written
    messagebox.askyesno = mb_yesno
    win.destroy()

    # q range boxes act immediately: arcs on the map, 'used' spots, results fitted with another range greyed
    q0, q1 = app.vars["q_min"].get(), app.vars["q_max"].get()
    app.vars["q_min"].set("0.35"); app.vars["q_max"].set("0.36"); app._on_qrange(); root.update()
    assert sum(1 for l in app.ax.lines if l.get_linestyle() == "--") == 2
    n_used = sum(1 for i in app.tv_spots.get_children("") if app.tv_spots.set(i, "used") == "yes")
    assert n_used < 6 and all("stale" in app.tv_res.item(i, "tags") for i in app.tv_res.get_children("")), n_used
    n_ev = len(app.evals)
    import tkinter.messagebox as mb
    warned = []
    mb.showwarning = lambda *a, **k: warned.append(a)
    app._run_check(); root.update()
    assert warned and len(app.evals) == n_ev and not app.busy        # too few spots: refused, nothing started
    app.vars["q_min"].set(q0); app.vars["q_max"].set(q1); app._on_qrange(); root.update()
    assert not any("stale" in app.tv_res.item(i, "tags") for i in app.tv_res.get_children(""))
    L0 = app.vars["sample_length_mm"].get()                        # sample length changes the window -> stale
    app.vars["sample_length_mm"].set("12"); app._on_qrange(); root.update()
    assert all("stale" in app.tv_res.item(i, "tags") for i in app.tv_res.get_children(""))
    assert abs(app._read_fields().q_rel() - 6.0 / app._read_fields().sdd_mm) < 1e-12
    app.vars["sample_length_mm"].set(L0); app._on_qrange(); root.update()
    assert not any("stale" in app.tv_res.item(i, "tags") for i in app.tv_res.get_children(""))

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
