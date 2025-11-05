#!/bin/bash

# ================================================================
# Script de Instalação Completa - ROS Noetic + Gazebo + Ardupilot
# ================================================================

set -e  # Sai se houver erro

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Funções de output colorido
function print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

function print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

function print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

function print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

function print_separator() {
    echo ""
    echo "=================================================================="
}

# ================================================================
# ETAPA 1: Configuração inicial
# ================================================================
print_separator
print_info "Iniciando configuração do ambiente..."
print_separator

# Vai para pasta home
cd ~

# Cria a variável ROS_WS
export ROS_WS=~/catkin_ws
print_info "Variável ROS_WS definida: $ROS_WS"

# Adiciona ROS_WS ao .bashrc se ainda não existir
if ! grep -q "export ROS_WS=~/catkin_ws" ~/.bashrc; then
    echo "export ROS_WS=~/catkin_ws" >> ~/.bashrc
    print_success "Variável ROS_WS adicionada ao .bashrc"
else
    print_warning "Variável ROS_WS já existe no .bashrc"
fi

# ================================================================
# ETAPA 2: Verificar se ROS já está instalado
# ================================================================
print_separator
print_info "Verificando instalação do ROS..."
print_separator

ROS_INSTALLED=false
if [ -f "/opt/ros/noetic/setup.bash" ]; then
    print_success "ROS Noetic já está instalado!"
    ROS_INSTALLED=true
    source /opt/ros/noetic/setup.bash
else
    print_warning "ROS Noetic não encontrado. Iniciando instalação..."
fi

# ================================================================
# ETAPA 3: Instalação do ROS Noetic (se necessário)
# ================================================================
if [ "$ROS_INSTALLED" = false ]; then
    print_separator
    print_info "Instalando ROS Noetic Desktop Full..."
    print_separator
    
    # Instalar lsb-release se necessário
    sudo apt update
    sudo apt install -y lsb-release software-properties-common
    
    # Adicionar repositórios
    print_info "Adicionando repositórios do ROS..."
    sudo apt-add-repository universe
    sudo apt-add-repository multiverse
    sudo apt-add-repository restricted
    
    # Setup sources.list
    sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
    
    # Setup keys
    sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
    
    # Atualizar apt
    print_info "Atualizando lista de pacotes..."
    sudo apt update
    
    # Instalar ROS Desktop Full
    print_info "Instalando ROS Noetic Desktop Full (isso pode demorar)..."
    sudo apt install -y ros-noetic-desktop-full
    
    # Instalar MAVROS
    print_info "Instalando MAVROS..."
    sudo apt install -y ros-noetic-mavros ros-noetic-mavros-extras
    
    # Instalar geographic lib datasets
    print_info "Instalando Geographic Lib datasets..."
    wget -q https://raw.githubusercontent.com/mavlink/mavros/master/mavros/scripts/install_geographiclib_datasets.sh
    sudo bash ./install_geographiclib_datasets.sh
    rm -f ./install_geographiclib_datasets.sh
    
    # Instalar rosdep
    print_info "Instalando e inicializando rosdep..."
    sudo apt install -y python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool python3-catkin-tools build-essential cmake
    
    if [ ! -f "/etc/ros/rosdep/sources.list.d/20-default.list" ]; then
        sudo rosdep init
    fi
    rosdep update
    
    # Adicionar source do ROS ao .bashrc
    if ! grep -q "source /opt/ros/noetic/setup.bash" ~/.bashrc; then
        echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
        print_success "Source do ROS adicionado ao .bashrc"
    fi
    
    source /opt/ros/noetic/setup.bash
    print_success "ROS Noetic instalado com sucesso!"
    
    # Criar workspace catkin se não existir
    if [ ! -d "$ROS_WS" ]; then
        print_info "Criando workspace catkin..."
        mkdir -p $ROS_WS/src
        cd $ROS_WS
        catkin_make
        print_success "Workspace catkin criado!"
    fi
    
    # Criar ambiente virtual Python
    if [ ! -d "$ROS_WS/src/venv" ]; then
        print_info "Criando ambiente virtual Python..."
        sudo apt install -y python3-venv python3-pip
        cd $ROS_WS/src
        python3 -m venv venv
        source venv/bin/activate
        pip install --upgrade pip
        pip install pymavlink MAVProxy future
        deactivate
        print_success "Ambiente virtual Python criado!"
    fi
    
    # Adicionar source do workspace ao .bashrc
    if ! grep -q "source $ROS_WS/devel/setup.bash" ~/.bashrc; then
        echo "source $ROS_WS/devel/setup.bash" >> ~/.bashrc
        print_success "Source do workspace adicionado ao .bashrc!"
    fi
    
    # Adicionar aliases ao .bashrc
    print_info "Adicionando aliases úteis ao .bashrc..."
    
    aliases=(
        "alias cw='cd \$ROS_WS'"
        "alias cs='cd \$ROS_WS/src'"
        "alias cm='cd \$ROS_WS && catkin_make'"
        "alias st='cd \$ROS_WS && source devel/setup.bash'"
        "source \$ROS_WS/src/venv/bin/activate"
        "export ROS_HOSTNAME=localhost"
        "export ROS_MASTER_URI=http://\${ROS_HOSTNAME}:11311"
    )
    
    for alias_cmd in "${aliases[@]}"; do
        if ! grep -q "$alias_cmd" ~/.bashrc; then
            echo "$alias_cmd" >> ~/.bashrc
            print_success "Adicionado: $alias_cmd"
        fi
    done
fi

# ================================================================
# ETAPA 4: Configuração do repositório privado e clonagem
# ================================================================
print_separator
print_info "Configurando repositórios do Gazebo..."
print_separator

# Configurações do repositório privado
GITHUB_USER="IagoBiundini"
GITHUB_TOKEN="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"

# Criar pasta src se não existir
if [ ! -d "$ROS_WS/src" ]; then
    print_info "Criando diretório src..."
    mkdir -p $ROS_WS/src
fi

cd $ROS_WS/src

# Clonar rover-argo-gazebo se não existir
if [ ! -d "$ROS_WS/src/rover-argo-gazebo" ]; then
    print_info "Clonando rover-argo-gazebo..."
    git clone -b main https://github.com/ttrindader/rover-argo-gazebo
    print_success "rover-argo-gazebo clonado!"
else
    print_warning "rover-argo-gazebo já existe, pulando clone..."
fi

# Atualizar e instalar dependências
print_info "Instalando dependências do sistema..."
sudo apt update
sudo apt install -y build-essential

# Instalar dependências do ROS
print_info "Instalando dependências do ROS com rosdep..."
cd $ROS_WS
rosdep install --from-paths . --ignore-src -r -y

# Adicionar GAZEBO_MODEL_PATH ao .bashrc
if ! grep -q "export GAZEBO_MODEL_PATH=\${GAZEBO_MODEL_PATH}:\$ROS_WS/src/rover-argo-gazebo/rover_argo_gazebo/models" ~/.bashrc; then
    echo 'export GAZEBO_MODEL_PATH=${GAZEBO_MODEL_PATH}:$ROS_WS/src/rover-argo-gazebo/rover_argo_gazebo/models' >> ~/.bashrc
    print_success "GAZEBO_MODEL_PATH adicionado ao .bashrc"
else
    print_warning "GAZEBO_MODEL_PATH já existe no .bashrc"
fi

source ~/.bashrc

# ================================================================
# ETAPA 5: Instalação do Livox Laser Simulation
# ================================================================
print_separator
print_info "Instalando Livox Laser Simulation..."
print_separator

sudo apt-get install -y libignition-math4-dev

cd $ROS_WS/src

if [ ! -d "$ROS_WS/src/livox_laser_simulation" ]; then
    print_info "Clonando livox_laser_simulation..."
    git clone https://github.com/Livox-SDK/livox_laser_simulation
    cd livox_laser_simulation
    sed -i 's/-std=c++11/-std=c++17/gi' CMakeLists.txt
    print_success "livox_laser_simulation clonado e configurado!"
else
    print_warning "livox_laser_simulation já existe, pulando clone..."
    cd livox_laser_simulation
fi

# ================================================================
# ETAPA 6: Instalação do ardupilot_gazebo
# ================================================================
print_separator
print_info "Instalando ardupilot_gazebo..."
print_separator

cd $ROS_WS/src

if [ ! -d "$ROS_WS/src/ardupilot_gazebo" ]; then
    print_info "Clonando ardupilot_gazebo..."
    git clone https://github.com/ttrindader/ardupilot_gazebo
    cd ardupilot_gazebo
    mkdir build && cd build
    print_info "Compilando ardupilot_gazebo..."
    cmake ..
    make -j4
    sudo make install
    print_success "ardupilot_gazebo instalado!"
else
    print_warning "ardupilot_gazebo já existe. Recompilando..."
    cd ardupilot_gazebo/build
    cmake ..
    make -j4
    sudo make install
fi

# Adicionar setup do Gazebo ao .bashrc
if ! grep -q "source /usr/share/gazebo-11/setup.bash" ~/.bashrc; then
    echo 'source /usr/share/gazebo-11/setup.bash' >> ~/.bashrc
    print_success "Setup do Gazebo-11 adicionado ao .bashrc"
fi

source ~/.bashrc
print_info "GAZEBO_PLUGIN_PATH: $GAZEBO_PLUGIN_PATH"

# ================================================================
# ETAPA 7: Instalação do Ardupilot
# ================================================================
print_separator
print_info "Instalando Ardupilot..."
print_separator

# Adicionar ARDUPILOT_PATH ao .bashrc
if ! grep -q "export ARDUPILOT_PATH=~/ardupilot" ~/.bashrc; then
    echo 'export ARDUPILOT_PATH=~/ardupilot' >> ~/.bashrc
    print_success "ARDUPILOT_PATH adicionado ao .bashrc"
fi

export ARDUPILOT_PATH=~/ardupilot
source ~/.bashrc

# Criar diretório do Ardupilot se não existir
if [ ! -d "$ARDUPILOT_PATH" ]; then
    mkdir -p $ARDUPILOT_PATH
fi

cd $ARDUPILOT_PATH

# Clonar Ardupilot se não existir
if [ ! -d "$ARDUPILOT_PATH/ardupilot" ]; then
    print_info "Clonando Ardupilot (branch roverCitenel)..."
    git clone https://github.com/lmhonorio/ardupilot -b roverCitenel
    cd ardupilot
    print_info "Atualizando submódulos..."
    git submodule update --init --recursive
    print_success "Ardupilot clonado!"
else
    print_warning "Ardupilot já existe, pulando clone..."
    cd ardupilot
fi

# Instalar pré-requisitos do Ardupilot
print_info "Instalando pré-requisitos do Ardupilot..."
./Tools/environment_install/install-prereqs-ubuntu.sh -y

print_info "Instalando pacotes Python adicionais..."
sudo pip install future pymavlink MAVProxy

# Adicionar completion e paths ao .bashrc
if ! grep -q "source \$ARDUPILOT_PATH/ardupilot/Tools/completion/completion.bash" ~/.bashrc; then
    echo 'source $ARDUPILOT_PATH/ardupilot/Tools/completion/completion.bash' >> ~/.bashrc
    print_success "Completion do Ardupilot adicionado ao .bashrc"
fi

if ! grep -q "export PATH=\$PATH:\$ARDUPILOT_PATH/ardupilot/Tools/autotest" ~/.bashrc; then
    echo 'export PATH=$PATH:$ARDUPILOT_PATH/ardupilot/Tools/autotest' >> ~/.bashrc
    print_success "Tools/autotest adicionado ao PATH"
fi

if ! grep -q "export PATH=/usr/lib/ccache:\$PATH" ~/.bashrc; then
    echo 'export PATH=/usr/lib/ccache:$PATH' >> ~/.bashrc
    print_success "ccache adicionado ao PATH"
fi

source ~/.bashrc

# Configurar Ardupilot
print_info "Configurando Ardupilot com waf..."
cd $ARDUPILOT_PATH/ardupilot
./waf configure

print_success "Ardupilot configurado!"

# ================================================================
# ETAPA 8: Finalização e build do workspace
# ================================================================
print_separator
print_info "Finalizando instalação..."
print_separator

cd $ROS_WS

# Executar prerequisites do rover-argo-gazebo
if [ -f "src/rover-argo-gazebo/rover_argo_gazebo/scripts/prerequisites.sh" ]; then
    print_info "Executando script de pré-requisitos do rover-argo-gazebo..."
    chmod +x src/rover-argo-gazebo/rover_argo_gazebo/scripts/prerequisites.sh
    ./src/rover-argo-gazebo/rover_argo_gazebo/scripts/prerequisites.sh
else
    print_warning "Script de pré-requisitos não encontrado, pulando..."
fi

# Build do workspace
print_info "Compilando workspace catkin (isso pode demorar)..."
catkin_make

# Source do workspace
source devel/setup.bash

# ================================================================
# RESUMO FINAL
# ================================================================
print_separator
print_success "INSTALAÇÃO COMPLETA CONCLUÍDA COM SUCESSO!"
print_separator
echo ""
print_info "Resumo da instalação:"
echo "  ✓ ROS Noetic Desktop Full instalado"
echo "  ✓ Workspace catkin criado em: $ROS_WS"
echo "  ✓ rover-argo-gazebo clonado"
echo "  ✓ livox_laser_simulation instalado"
echo "  ✓ ardupilot_gazebo instalado"
echo "  ✓ Ardupilot (roverCitenel) instalado em: $ARDUPILOT_PATH"
echo "  ✓ Todas as dependências instaladas"
echo "  ✓ Workspace compilado com sucesso"
echo ""
print_warning "IMPORTANTE: Execute 'source ~/.bashrc' ou abra um novo terminal para que todas as mudanças tenham efeito!"
echo ""
print_info "Aliases disponíveis:"
echo "  cw  - navegar para \$ROS_WS"
echo "  cs  - navegar para \$ROS_WS/src"
echo "  cm  - navegar para \$ROS_WS e executar catkin_make"
echo "  st  - navegar para \$ROS_WS e carregar setup.bash"
echo ""
print_info "Para testar a instalação:"
echo "  1. Abra um novo terminal"
echo "  2. Execute: roscore"
echo "  3. Em outro terminal, execute: gazebo"
echo ""
print_success "Instalação finalizada! Aproveite o seu ambiente ROS + Gazebo + Ardupilot!"
print_separator

