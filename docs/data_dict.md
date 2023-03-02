# De todos os missings que a gente tem...

# Aprofundando o entendimento das variáveis

A ideia é registrar aqui informações sobre as variáveis para ser possível tomar melhores decisões para melhorar nosso modelo de previsão.

## CO_MUN_NOT ou ID_MUNICIP (unit_municipio_code_ibge)

Código do município pelo IBGE onde está localizada a **unidade** que registrou a notificação.

## CO_REGIONA ou ID_REGIONA (unit_regionais_saude_code_ibge)

Código das regionais de saúde dos municípios de notificação. Logo, precisamos ter a indicação do Município para que seja possível identificar a Regional de Saúde.

Uma região de saúde é espaço geográfico contínuo constituído por agrupamentos de Municípios limítrofes, delimitado a partir de identidades culturais, econômicas e sociais e de redes de comunicação e infraestrutura de transportes compartilhados, com a finalidade de integrar a organização e o planejamento de ações e serviços de saúde.

Pode ser de:

- Atenção primária (I)
- Urgência e emergência (II)
- Atenção Psicossocial (III)
- Atenção ambulatorial especializada e hospitalar (IV)
- Vigilância em saúde

## SG_UF_NOT (unit_uf_code_ibge)

Unidade Federativa onde está localizada a Unidade que realizou a notificação.

## CS_SEXO (patient_gender)

Pode ser:

- 1: masculino
- 2: feminino
- 9: ignorado

## COD_IDADE (não consta no documento)

**Analisar e verificar os dados que temos aqui**

## CS_GESTANT (patient_gest_period)

Idade gestacional do paciente

Aqui vai ser muito interessante analisar algumas coisinhas:

- Temos missings aqui?
- Se `patient_gender` == masculino, e `age` <= 9 => 6 (Não se aplica)
- Se `patient_gender` == feminino, e `age` >= 9 => Não deveríamos observar 6 (não se aplica)

## CS_RACA (patient_color)

Cor ou raça declarada pelo paciente:

- 1: Branca
- 2: Preta
- 3: Amarela
- 4: Parda
- 5: Indígena
- 9: Ignorado

Pode ser interessante para verificar algum ponto de viés no modelo gerado

## CS_ESCOL_N (patient_education)

Escolaridade do paciente.

- `age` < 7, preenchido automaticamente com "não se aplica"
- `age` < 7, não devemos encontrar o "não se aplica"

Temos as opções:

- 0: Sem escolaridade/analfabeto
- 1: Fundamental I
- 2: Fundamental II
- 3: Médio
- 4: Superior
- 5: Não se aplica
- 9: Ignorado

## SG_UF (patient_uf)

O preenchimento do CEP faz com que esse campo seja automaticamente preenchido. Podemos verificar isso!

Temos ao todo 31 códigos, isso também pode ser verificado.

## CS_ZONA (patient_zone_residence)

Zona geográfica de residência do paciente.

- 1: Urbana
- 2: Rural
- 3: Periurbana
- 9: Ignorado

## SURTO_SG (não consta no documento)

**Analisar e verificar os dados que temos aqui**

## NOSOCOMIAL (is_nosocomial)

Caso de SRAG com infecção adquirida após internação.

- 1: Sim
- 2: Não
- 9: Ignorado

Quando temos 1 nesse campo, é permitido digitar _data de início dos sintomas posterior a data de internação_.

## AVE_SUINO (is_ave_suino)

Paciente trabalha ou tem contato direto com aves, suínos, ou outro animal?

- 1: Sim
- 2: Não
- 9: Ignorado

## FEBRE (is_febre)

Paciente apresentou febre?

- 1: Sim
- 2: Não
- 9: Ignorado

## TOSSE (is_tosse)

Paciente apresentou tosse?

- 1: Sim
- 2: Não
- 9: Ignorado

## GARGANTA (is_garganta)

Paciente apresentou dor de garganta?

- 1: Sim
- 2: Não
- 9: Ignorado

## DISPNEIA (is_dispneia)

Paciente apresentou dispneia?

- 1: Sim
- 2: Não
- 9: Ignorado

## DESC_RESP (is_respiration_down)

Paciente apresentou desconforto respiratório?

- 1: Sim
- 2: Não
- 9: Ignorado

## SATURACAO (is_o2_saturation_down)

Paciente apresentou saturação 02 < 95%?

- 1: Sim
- 2: Não
- 9: Ignorado

## DIARREIA (is_diarreia)

Paciente apresentou diarréia?

- 1: Sim
- 2: Não
- 9: Ignorado

## VOMITO (is_vomito)

Paciente apresentou vômito?

- 1: Sim
- 2: Não
- 9: Ignorado

## OUTRO_SIN (is_outro_sinal)

Paciente apresentou outro sintomas?

- 1: Sim
- 2: Não
- 9: Ignorado

## OUTRO_DES (outro_sinal_desc)

Campo habilitado quando é selecionado 1 no `is_outro_sinal`

## PUERPERA (is_puerpera)

Mulher é puérpera ou parturiente (mulher que pariu recentemente - até 45 dias de parto)?

Habilitado somente em casos de mulher (`patient_gender` == 2)

## FATOR_RISC (is_fator_risco)

Paciente apresenta algum fator de risco?

- 1: Sim
- 2: Não
- 9: Ignorado

## CARDIOPATI (is_cardio_down_fr)

Paciente possui doença cardiovascular crônica?

- 1: Sim
- 2: Não
- 9: Ignorado

## HEMATOLOGI (is_hemato_down_fr)

Paciente possui doença Hematológica crônica?

## SIND_DOWN (is_sindrome_down_fr)

Paciente possui Síndrome de Down?

- 1: Sim
- 2: Não
- 9: Ignorado

## HEPATICA (is_hepatica_down_fr)

Paciente possui alguma doença hepática crônica?

- 1: Sim
- 2: Não
- 9: Ignorado

## ASMA (is_asma_fr)

Possui asma?

- 1: Sim
- 2: Não
- 9: Ignorado

## DIABETES (is_diabetes_fr)

Possui diabetes?

- 1: Sim
- 2: Não
- 9: Ignorado

## NEUROLOGIC (is_neuro_down_fr)

Possui doença neurológica?

- 1: Sim
- 2: Não
- 9: Ignorado

## PNEUMOPATI (is_pneumopatia_down_fr)

Possui pneumopatia?

- 1: Sim
- 2: Não
- 9: Ignorado

## IMUNODEPRE (is_imunodef_down_fr)

Possui imunodeficiência ou imunodepressão?

- 1: Sim
- 2: Não
- 9: Ignorado

## RENAL (is_renal_down_fr)

Possui doença crônica renal?

- 1: Sim
- 2: Não
- 9: Ignorado

## OBESIDADE (is_obesidade_down_fr)

Possui obesidade?

- 1: Sim
- 2: Não
- 9: Ignorado

## OBES_IMC (obesidade_imc)

Habilitado apenas se `is_obesidade_down_fr` == True

- 1: Sim
- 2: Não
- 9: Ignorado

## OUT_MORBI (is_outros_fr)

Possui outros fatores de risco?

- 1: Sim
- 2: Não
- 9: Ignorado

## MORB_DESC (outros_fr_desc)

Descrição dos outros fatores de risco caso o paciente tenha outros fatores de risco.

## VACINA (is_vacina)

Recebeu vacina contra Gripe na última campanha?

- 1: Sim
- 2: Não
- 9: Ignorado

Tomar a vacina de Gripe da última campanha fornece imunidade contra o SRAG?

## MAE_VAC (is_mae_vacina)

A mãe recebeu vacina? De que?

Habilitado apenas se a _idade for menor que 6 meses._

- 1: Sim
- 2: Não
- 9: Ignorado

## M_AMAMENTA (is_mae_amamenta)

A mãe amamenta a criança?

Habilitado apenas se a _idade for menor que 6 meses._

- 1: Sim
- 2: Não
- 9: Ignorado

## ANTIVIRAL (is_antiviral)

Usou antiviral para gripe?

- 1: Sim
- 2: Não
- 9: Ignorado

## TP_ANTIVIR (qual_antiviral)

Qual o outro antiviral?

- 1: Oseltamivir
- 2: Zanamivir
- 3: Outro

**Nem OUT_ANTIVIR nem DT_ANTIVIR tem nos dados!, seriam relacionados a essa variável.**

## HOSPITAL (is_internado)

Foi internado?

- 1: Sim
- 2: Não
- 9: Ignorado

## UTI (is_internado_uti)

Internado na UTI?

- 1: Sim
- 2: Não
- 9: Ignorado

Acredito que se o paciente não foi internado, ele tbm não foi internado na UTI. Vale a verificação.

## SUPORT_VEN (is_suporte_ventilador)

Usou suporte ventilatório?

- 1: Sim, invasivo
- 2: Sim, não invasivo
- 3: Não
- 9: Ignorado

## RAIOX_RES (x_ray_torax)

Resultado do Raio X do Toráx

- 1: Normal
- 2: Infiltrado intersticial
- 3: Consolidação
- 4: Misto
- 5: Outro
- 6: Não realizado
- 9: Ignorado

## RAIOX_OUT (x_ray_outro_desc)

Informar o resultado do RX de tórax caso Outros tenha sido selecionado.

Habilitado apenas se `x_ray_torax` set to 5.

## AMOSTRA (is_amostragem)

Coletou amostra para teste diagnóstico?

- 1: Sim
- 2: Não
- 9: Ignorado

## TP_AMOSTRA (tipo_amostragem)

Caso tenha sido coletado amostra (`is_amostragem` == 1), qual o tipo?

- 1: Secreção de Nasoorofaringe
- 2: Levado Broco-alveolar
- 3: Tecido post-mortem
- 4: Outro, qual?
- 5: LCR
- 9: Ignorado

## OUT_AMOST (amostragem_outros)

Se foi feito um outro tipo de amostragem (`tipo_amostragem` == 4), descreva?

## HISTO_VGM (não consta no documento)

**Analisar e verificar os dados que temos aqui**

## PAC_COCBO e PAC_DSCBO são iguais (patient_occupation)

Ocupação profissional do paciente. Código tabelado

## OUT_ANIM (patient_work_animals)

Habilitado coso 3 no AVE_SUINO. \*\*Buscar por caso 3 no ave_suino.

## DOR_ABD (is_dor_abdominal)

Apresentou dor abdominal?

- 1: Sim
- 2: Não
- 9: Ignorado

## FADIGA (is_fadiga)

Apresentou fadiga?

- 1: Sim
- 2: Não
- 9: Ignorado

## PERD_OLFT (is_olfato_loose)

Apresentou perda de olfato?

- 1: Sim
- 2: Não
- 9: Ignorado

## PERD_PALA (is_paladar_loose)

Apresentou perda de paladar?

- 1: Sim
- 2: Não
- 9: Ignorado

## TOMO_RES (results_tomografia)

Resultados da Tomografia

- 1: Tipico Covid-19
- 2: Indeterminado covid-19
- 3: Atipico covid-19
- 4: Negativo para Pneumonia
- 5: Outro
- 6: Não realizado
- 9: Ignorado

## TOMO_OUT (results_tomografia_outro)

No caso de outro, descreva o resultado

## VACINA_COV (is_vacina_covid)

Recebeu vacina covid-19?

- 1: Sim
- 2: Não
- 9: Ignorado

## DOSE_1_COV (is_vacina_covid_dose_1)

Tomou a primeira dose?

- 1: Sim
- 2: Não
- 9: Ignorado

## DOSE_2_COV (is_vacina_covid_dose_2)

Tomou a segunda dose?

- 1: Sim
- 2: Não
- 9: Ignorado

## FNT_IN_COV (data_integration)

Manual ou Integração. Forma de como os dados foram coletados.

- 1: Manual
- 2: Integração

## TP_IDADE (age_type)

Tipo idade. Contabilizada da data de nascimento até a data dos sintomas.

- 1: Dia
- 2: Mês
- 3: Ano

## SEM_NOT (semana_epid_notification)

Semana Epidemiológica do preenchimento da ficha de notificação. Calculado a partir da data dos Primeiros Sintomas.

## SEM_PRI (semana_epid_sintomas)

Semana Epidemiológica dos primeiros sintomas. Calculado a partir da data dos Primeiros Sintomas.

## NU_IDADE_N (patient_age)

Idade informada pelo paciente. Na falta dessa informação o sistema adiciona uma data aparente.

Idades devem ser <= 150.

## CLASSI_FIN (target)

Nosso target!

- 1: SRAG por influenza
- 2: por outro vírus
- 3: por outro agente
- 4: por outro n especificado
- 5: por convid-19
