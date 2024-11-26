##########################################################
## ROBO TRADER - MEDIAS MOVEIS - OPERAÇÃO EM TEMPO REAL ##

import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keyboard
import MetaTrader5 as mt5
import talib

import time
import warnings

# Supress specific RuntimeWarning
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in find_crossover")

# Configurações do gráfico
plt.ion()  # Ativar modo interativo

# Variáveis globais
symbol = "WINZ24"
taxa_point = 0.2

time_frame = mt5.TIMEFRAME_M5
Fast = 1
Slow = 3
stop_loss = 400
stop_gain = 1500
point = None
status_operation = "FECHADA"
order_type = '0'

Performance = pd.DataFrame(columns=[
                                    'Numero', 
                                    'MT5', 
                                    'Status', 
                                    'Ativo', 
                                    'Dia', 
                                    'Hora Abertura', 
                                    'Hora Fechamento', 
                                    'Tempo de Operacao', 
                                    'Qtde', 
                                    'Lado', 
                                    'Preco de entrada', 
                                    'Preco de saida', 
                                    'Resultado Intervalo', 
                                    'Resultado %', 
                                    'Resultado Financeiro', 
                                    'Resultado Acumulado'
                                ])

def initialize_mt5():
    if not mt5.initialize():
        print("initialize() failed")
        mt5.shutdown()
        return False
    return True

def configure_symbol(symbol):
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print("O ativo especificado não foi encontrado")
        return False
    
    if not symbol_info.visible:
        print(symbol, "não está visível, tentando alterar...")
        if not mt5.symbol_select(symbol, True):
            print("Falha ao selecionar", symbol)
            return False
        
    global point
    point = symbol_info.point
    return True

def Indicators_data(data, Fast, Slow):
    data['sma_slow'] = data['Price'].rolling(Slow).mean()
    data['sma_fast'] = data['Price'].ewm(span=Fast).mean()
    data['RSI'] = talib.RSI(data['Price'], timeperiod=14)
    return data

def find_crossover(fast_sma, slow_sma):
    if fast_sma > slow_sma:
        return 'BUY'
    elif fast_sma < slow_sma:
        return 'SELL'
    else:
        return 'HOLD'

def get_current_time():
    while True:
        now = dt.datetime.now()
        yield now
        next_second = (now + dt.timedelta(seconds=1)).replace(microsecond=0)
        sleep_duration = (next_second - dt.datetime.now()).total_seconds()
        time.sleep(abs(sleep_duration))

def go_order(symbol, entry_time, entry_price, stop_loss, stop_gain, lado):
    
    lot = 1.0
    deviation = 20
    stop_loss = round(entry_price - stop_loss * point, 0) + 150 # stop loss de emergência
    stop_gain = round(entry_price + stop_gain * point, 0) + 150
    order_type = mt5.ORDER_TYPE_BUY if (lado == 'Compra') else mt5.ORDER_TYPE_SELL
    
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": order_type,
        "price": entry_price,
        "sl": stop_loss,
        "tp": stop_gain,
        "deviation": deviation,
        "magic": 234000,
        "comment": "python script open",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_RETURN,
    }

    result = mt5.order_send(request)
    check_mt5 = result #"Ordem enviada com sucesso" if (result.retcode == 10009) else f'Falha ao enviar ordem - código: {result}'

    Operation = len(Performance) + 1
    Status = 'Aberta'

    new_row = [
        Operation,      
        check_mt5, 
        Status, 
        symbol, 
        dt.datetime.now().strftime('%d/%m/%Y'),
        pd.to_datetime(entry_time, unit='s'), 
        None, 
        None, 
        lot, 
        lado, 
        entry_price, 
        None, 
        None, 
        None, 
        None, 
        None
        ]
    
    Performance.loc[Operation] = new_row
    return Performance

def out_order(symbol, out_price, out_time):

    lot = 1.0
    deviation = 20
    order_type = mt5.ORDER_TYPE_SELL if (Performance.loc[Performance['Status'] == 'Aberta', 'Lado'].iloc[0] == 'Compra') else mt5.ORDER_TYPE_BUY 
    
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": order_type,
        "price": out_price,
        "deviation": deviation,
        "magic": 234000,
        "comment": "python script open",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_RETURN,
    }
    
    result = mt5.order_send(request)
    
    check_mt5 = result #"Ordem enviada com sucesso" if (result.retcode == 10009) else f'Falha ao enviar ordem - código: {result.retcode}'
    
    Performance.loc[Performance['Status'] == 'Aberta', 'MT5'] = check_mt5
    Performance.loc[Performance['Status'] == 'Aberta', 'Hora Fechamento'] = [pd.to_datetime(out_time, unit='s')]
    # Atualizar 'Resultado Intervalo' de forma condicional para todas as linhas com 'Status' igual a 'ABERTA'
    Resultado = np.where(
        Performance.loc[Performance['Status'] == 'ABERTA', 'Lado'] == 'Compra',
        Performance.loc[Performance['Status'] == 'ABERTA', 'Preco de saida'] - Performance.loc[Performance['Status'] == 'ABERTA', 'Preco de entrada'],
        Performance.loc[Performance['Status'] == 'ABERTA', 'Preco de entrada'] - Performance.loc[Performance['Status'] == 'ABERTA', 'Preco de saida']
    )
    
    # Aplicar o resultado na coluna 'Resultado Intervalo'
    Performance.loc[Performance['Status'] == 'ABERTA', 'Resultado Intervalo'] = Resultado
    
    Performance.loc[Performance['Status'] == 'Aberta', 'Resultado %'] = ((out_price / Performance['Preco de entrada']) - 1) * 100
    Performance.loc[Performance['Status'] == 'Aberta', 'Resultado Financeiro'] = Performance['Resultado Intervalo'] * taxa_point
    Performance.loc[Performance['Status'] == 'Aberta', 'Status'] = 'Fechada'

    Performance['Hora Abertura'] = pd.to_datetime(Performance['Hora Abertura'])
    Performance['Hora Fechamento'] = pd.to_datetime(Performance['Hora Fechamento'])
    Performance['Tempo de Operacao'] = Performance['Hora Fechamento'] - Performance['Hora Abertura']
    
    Performance['Resultado Acumulado'] = Performance['Resultado Financeiro'].cumsum()
    
    return Performance

def save_day(data, Performance, fig_path):
    
    with pd.ExcelWriter(f'C:/Users/paulo/Desktop/UFSM - 2024/Projeto/Outputs/Trading_{dt.datetime.now().strftime("%d%m-%H%M%S")}.xlsx', engine='xlsxwriter') as writer:
        
        data.to_excel(writer, sheet_name='Data', index=False)
        Performance.to_excel(writer, sheet_name='Performance', index=False)
        workbook = writer.book
        worksheet = workbook.add_worksheet('Chart')
        writer.sheets['Chart'] = worksheet
        worksheet.insert_image('B2', fig_path)
        


def update_plot(data, symbol):
    plt.clf()
    plt.plot(data.index, data['Price'], label='Preço', color='k')
    plt.plot(data.index, data['sma_fast'], color='b', label='Média Rápida')
    plt.plot(data.index, data['sma_slow'], color='g', label='Média Lenta')
    plt.xticks(data.index[::5], labels=data['Time'].dt.strftime('%d | %H:%M')[::5], rotation=45)
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.plot(data[data['Status'] == 1].index, data['Price'][data['Status'] == 1], '^', markersize=15, color='g', label='Buy')
    plt.plot(data[data['Status'] == -1].index, data['Price'][data['Status'] == -1], 'v', markersize=15, color='r', label='Sell')
    plt.title('Algo Trading', fontsize=20)

    if status_operation == 'ABERTA':
        texto = f'Resultado operação: {Resultado} pts'
    elif len(Performance) > 0:
        texto = f'Resultado Acumulado($): {Performance["Resultado Acumulado"].iloc[-1]}'
    else:
        texto = 'Sem operação'
    
    plt.text(0.02, 0.90, texto, transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.9))

    rsi_value = data['RSI'].iloc[-1] 
    if  rsi_value > 60:
        texto_rsi = f'RSI Index: {rsi_value:.1f} - Sobrecomprado'
    elif rsi_value < 45:
        texto_rsi = f'RSI Index: {rsi_value:.1f} - Sobrevendido'
    else:
        texto_rsi = f'RSI Index: {rsi_value:.1f}'
    plt.text(0.02, 0.95, f"{texto_rsi}", transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.9))
    
    plt.pause(0.01)

def main():
    
    if not initialize_mt5():
        return
    if not configure_symbol(symbol):
        mt5.shutdown()
        return

    data = pd.DataFrame(columns=['Time', 'Price'])
    Ticks = mt5.copy_rates_from(symbol, time_frame, time.time(), 50)
    data['Price'] = Ticks['close']
    data['Time'] = pd.to_datetime(Ticks['time'], unit='s')
    data['Price'] = pd.to_numeric(data['Price'])
    data['Status'] = 0

    data = Indicators_data(data, Fast, Slow)
    data['Recomendação'] = np.vectorize(find_crossover)(data['sma_fast'], data['sma_slow'])
    data['Signal'] = np.where(data['sma_fast'] > data['sma_slow'], 1, 0)
    data['Position'] = data['Signal'].diff()

    entry_time = None
    global status_operation
    global Resultado
    time_generator = get_current_time()

    while True:
        time_str = next(time_generator)
        min = time_str.strftime('%M')
        seg = time_str.strftime('%S')
        tick = mt5.symbol_info_tick(symbol)

        if tick is None:
            print("Falha ao obter informações do tick", symbol)
            break
        
        last = data['Time'].iloc[-1]
        min_last = last.minute

        if min_last % 5 != 0 and int(min) % 5 == 0 and int(seg) == 0:
                        
            new_row = {'Time': time_str, 'Price': tick.last}
            data = data._append(new_row, ignore_index=True)

            data.at[data.index[-2], 'Time'] = time_str #pd.to_datetime(tick.time, unit='s')
            data.at[data.index[-2], 'Price'] = tick.last 

            data = Indicators_data(data, Fast, Slow)
            data['Recomendação'] = np.vectorize(find_crossover)(data['sma_fast'], data['sma_slow'])
            data['Signal'] = np.where(data['sma_fast'] > data['sma_slow'], 1, 0)
            data['Position'] = data['Signal'].diff()

            if status_operation == 'FECHADA':
                
                #### Buy Order - ENTRADA
                
                if data['Position'].iloc[-2] == 1 and data['Signal'].iloc[-2] == 1 and data['RSI'].iloc[-2] < 100 : # MELHOR 45
                
                    status_operation = 'ABERTA'
                    order_type = 'Compra'
                    entry_price = tick.ask
                    entry_time = tick.time
                    data.at[data.index[-2], 'Status'] = 1
                    print(f"Ordem de compra: Preço:{entry_price} horário: {entry_time}")
                    Performance = go_order(symbol, entry_time, entry_price, stop_loss, stop_gain, 'Compra')
                    print(f"Resposta mt5: {Performance['MT5'].iloc[-1]}")

                #### Sell order - ENTRADA
                
                elif data['Position'].iloc[-2] == -1 and data['Signal'].iloc[-2] == 0 and data['RSI'].iloc[-2] > 0 : # MELHOR 60
                    
                    status_operation = 'ABERTA'
                    order_type = 'Venda'
                    entry_price = tick.bid
                    sell_time = tick.time
                    data.at[data.index[-2], 'Status'] = -1
                    Performance = go_order(symbol, entry_time, entry_price, stop_loss, stop_gain, 'Venda')
                    print(f"Resposta mt5: {Performance['MT5'].iloc[-1]}")
            
            elif status_operation == 'ABERTA':
                
                # BUY SIDE - SAIDA
                
                if data['Position'].iloc[-2] == -1 and data['Signal'].iloc[-2] == 0 and order_type == 'Compra':
                    
                    status_operation = 'FECHADA'
                    out_price = tick.ask
                    out_time = tick.time
                    data.at[data.index[-2], 'Status'] = -1
                    Performance = out_order(Performance, out_price, out_time)
                    print(f"Resposta mt5: {Performance['MT5'].iloc[-1]}")
            

                # SEL SIDE - SAIDA
                
                elif data['Position'].iloc[-2] == 1 and data['Signal'].iloc[-2] == 1 and order_type == 'Venda':
                    
                    out_price = tick.bid
                    out_time = tick.time
                    status_operation = 'FECHADA'
                    data.at[data.index[-2], 'Status'] = 1
                    Performance = out_order(Performance, out_price, out_time)
                    print(f"Resposta mt5: {Performance['MT5'].iloc[-1]}")
            
        else:
            
            data.at[data.index[-1], 'Time'] = time_str #pd.to_datetime(tick.time, unit='s')
            data.at[data.index[-1], 'Price'] = tick.last
            data = Indicators_data(data, Fast, Slow)
            data['Recomendação'] = np.vectorize(find_crossover)(data['sma_fast'], data['sma_slow'])

        if status_operation == 'ABERTA':
            
            Resultado = (tick.last - entry_price) if order_type == 'Compra' else (entry_price - tick.last)
            
            #print(f'Gain: {Resultado}' if Resultado > 0 else f'Loss: {Resultado}')

            if Resultado <= -stop_loss or Resultado >= stop_gain:

                status_operation = "FECHADA"
                out_price = tick.bid if order_type == 'Compra' else tick.ask
                out_time = tick.time
                data.at[data.index[-1], 'Status'] = -1
                Performance = out_order(symbol, out_price, out_time)
                print(f'Resposta mt5: {Performance["MT5"].iloc[-1]}')

        if pd.to_datetime(tick.time, unit='s').hour == 18:

            if status_operation == 'ABERTA':

                status_operation = "FECHADA"
                out_price = tick.bid if order_type == 'Compra' else tick.ask
                out_time = tick.time
                data.at[data.index[-1], 'Status'] = -1
                Performance = out_order(symbol, out_price, out_time)
                print(f'Resposta mt5: {Performance["MT5"].iloc[-1]}')

            print("Fim do Expediente")

            fig_path = f'C:/Users/paulo/Desktop/UFSM - 2024/Projeto/Outputs/Figure_{dt.datetime.now().strftime("%d%m-%H%M%S")}.png'
            plt.savefig(fig_path)
            save_day(data, Performance, fig_path)
            break

        update_plot(data, symbol)

        if keyboard.is_pressed('F12'):

            print("Tecla F12 pressionada")
            fig_path = f'C:/Users/paulo/Desktop/UFSM - 2024/Projeto/Outputs/Figure_{dt.datetime.now().strftime("%d%m-%H%M%S")}.png'
            plt.savefig(fig_path)
            save_day(data, Performance, fig_path)

            break

    plt.ioff()
    plt.show()
    mt5.shutdown()

if __name__ == "__main__":
    main()
