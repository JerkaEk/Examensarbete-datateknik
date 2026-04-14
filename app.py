import logging
from core.controller import Controller

def main():
  logging.basicConfig(
    level=logging.DEBUG,
    format="(asctime)s [%(levelname)s %(name)s: %(message)s"
  )

  controller = Controller()
  controller.run()

if __name__ == "__main__":
  main()
