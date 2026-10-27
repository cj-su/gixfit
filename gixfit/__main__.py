from .cli import main

if __name__ == "__main__":      # guard: worker processes (null tests) re-import this module on Windows
    main()
