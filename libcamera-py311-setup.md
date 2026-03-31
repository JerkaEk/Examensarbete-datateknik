# Setup guide MediaPipe + picamera2 for Raspberry Pi5(bookworm)

## The issue
- picamera2 runs with system level python build, currently Python 3.13 (Compiled libcamera .so)
- MediaPipe currently doesn't have support for Python > 3.11 

This guide solves this by re-compiling the libcamera binding to Python 3.11 and link it to venv.

### Step 1 - Intall build dependencies
Run the following commands:
	sudo apt update
	sudo apt install -y \
	  build-essential libssl-dev zlib1g-dev libbz2-dev \
	  libreadline-dev libsqlite3-dev libffi-dev liblzma-dev \
	  libgdbm-dev libncurses-dev tk-dev

### Step 2 - Build Python 3.11.9
This can be done in various ways, pyenv was the chosen solution for this case.

### Step 3 - Rebuild libcamera binding to Python 3.11
Run the following commands:
	# Install libcamera build dependencies
	sudo apt install -y \
	  meson ninja-build pkg-config \
	  libcamera-dev libgnutls28-dev libevent-dev \
	  libyaml-dev python3-yaml python3-jinja2 python3-ply \
	  libglib2.0-dev libgstreamer1.0-dev \
	  pybind11-dev
	
	# Clone Raspberry libcamera fork
	cd /src	# Or whatever directory you prefer
	git clone --depth 1 https://github.com/raspberrypi/libcamera.git
	cd libcamera

	# Configurate with Python 3.11
	meson setup build \
	  -Dpycamera=enabled \
	  -Dpipelines=rpi/vc4,rpi/pisp \
	  -Dprefix=/opt/libcamera-py311 \
	  -Dpython.platlibdir=lib \
	  -Dpython.purelibdir=lib \
	  --pkg-config-path=/usr/lib/aarch64-linux-gnu/pkgconfig	

The meso should be able to automatically locate /opt/python3.11/bin/python3.11
If it fails, explicity state -Dpython=/opt/python3.11/bin/python3.11

	ninja -C build
	
	# Locate the .so-file:
	find build -name "_libcamera*.so" -print

### Step 4 - Create venv with python 3.11.9 and install the libcamera binding to the env
	# The venv creation is only relevant if you installed Python with pyenv
	pyenv local 3.11
	python3 -m venv venv
	pip install --upgrade pip

	# Copy the new .so-file
	SITE=$(python -c "import site; print(site.getsitepackages()[0])")
	mkdir -p "$SITE/libcamera"

	# Navigate to directory you cloned libcamera to and copy the libcamera Python files
	cp libcamera/build/src/py/libcamera/*.so "$SITE/libcamera/"
	cp libcamera/build/src/py/libcamera/*.py "$SITE/libcamera/"
	cp libcamera/src/py/libcamera/__init__.py "$SITE/libcamera/"

	# Verify
	# Verifiera
	python -c "import libcamera; print('libcamera OK:', dir(libcamera))"

### Step 5 - Install dependencies
	source venv/bin/activate
	# Installing picamera without apt-dependencies since they are already installed in the system
	pip install picamera2 --no-deps
	# The rest can be installed with the requirements.txt
	pip install -r requirements.txt

## Step 6 - Create pykms dummy
pykms is a part of the kmsxx projects and not availble for PyPI.
Since DRM-preview is not needed for the application we'll create a dummy for picamera2 to import and avoid having it crash.ä
	
	SITE=$(python -c "import site; print(site.getsitepackages()[0])")

	mkdir -p "$SITE/kms" "$SITE/pykms"

	cat > "$SITE/kms/__init__.py" << 'EOF'
	"""Stub for kms/pykms — provides enough surface for picamera2 to import.
	DRM preview will not function, but frame capture works fine."""

	class _FakeEnum:
        """Returns a dummy value for any attribute access."""
        def __getattr__(self, name):
        return name

	PixelFormat = _FakeEnum()
	DrmConnector = type('DrmConnector', (), {})
	DrmCrtc = type('DrmCrtc', (), {})
	DrmFrameBuffer = type('DrmFrameBuffer', (), {})
	DrmPlane = type('DrmPlane', (), {})
	Card = type('Card', (), {})
	AtomicReq = type('AtomicReq', (), {})
	EOF
	
	# Create kms dummy as well since picamera2 uses it as fallback.
	cp "$SITE/kms/__init__.py" "$SITE/pykms/__init__.py"

### Step 7 - Verify
Step 5 and 6 was is not fully documented so there might be some dependencies missing but the verifications should state those.
This will be updated in the near future to fully cover anything missing

	# Verify
	python -c "import mediapipe; print('MediaPipe', mediapipe.__version__)"
	python -c "from picamera2 import Picamera2; print('picamera2 OK')"
