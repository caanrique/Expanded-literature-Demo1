import os
import pickle
import gc
import torch
from sentence_transformers import SentenceTransformer, util
from llama_cpp import Llama
import gradio as gr
import re  # Asegúrate de tenerlo al inicio con los otros imports
import random 

def limpiar_respuesta(respuesta):
    """Elimina estructuras JSON que a veces genera el modelo."""
    # Patrón para [{'text': '...', 'type': 'text'}]
    match = re.search(r"\[\{'text': '(.*?)', 'type': 'text'\}\]", respuesta)
    if match:
        return match.group(1)
    # Si no hay JSON, devolvemos la respuesta tal cual
    return respuesta


# Activar descarga rápida
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# =============================================================================
# 📚 DATOS DE LOS CUENTOS (mismos textos, pon los completos)
# =============================================================================
CUENTOS = {
    "corazon_delator": {
        "titulo": "The Tell-Tale Heart",
        "autor": "Edgar Allan Poe",
        "personaje": "Narrator (paranoid killer)",
        "texto": """Verdaderamente, estoy nervioso, muy, muy terriblemente nervioso, lo he estado y lo soy; pero ¿por qué decís que estoy loco? La enfermedad había agudizado mis sentidos, no los había destruido, no los había embotado. Sobre todo, el sentido del oído se había vuelto agudo. Oía todas las cosas del cielo y de la tierra. Oía muchas cosas del infierno. ¿Cómo, pues, estoy loco? Escuchen y observen con qué salud de mente, con qué serenidad puedo referirles toda la historia.

No puede atribuirse a una pasión. Amaba al viejo. Jamás me había hecho mal alguno. Jamás me había insultado. Yo no codiciaba su oro. Creo que era su ojo. Sí, era esto. Tenía el ojo de un buitre, un ojo azul pálido, con una película sobre él. Siempre que caía sobre mí, mi sangre se helaba; y así, poco a poco, muy gradualmente, decidí quitarme la vida del viejo y, de este modo, liberarme del ojo para siempre.

Y he aquí ahora el primer ensayo de mi astucia. Nunca ha sido tan amable con el viejo como lo fui durante toda la semana anterior a su asesinato. Por las noches, cuando abría su puerta, le daba una vuelta al cerrojo, encendía mi linterna, me asomaba a su habitación y le dirigía el rayo de luz directamente al ojo. Y todo esto, con la mayor cautela posible. Porque no quería despertar al viejo. Al cabo de una hora, tal vez, cerraba cuidadosamente la puerta y volvía a mi habitación. Y por las mañanas, al amanecer, entraba con audacia en su habitación y le hablaba con voz cordial, llamándole por su nombre con afecto y preguntándole cómo había pasado la noche. Porque nunca podría adivinar que cada noche, precisamente a las doce, yo miraba dentro de su habitación mientras él dormía.

Al octavo día entré más temprano que de costumbre. El viejo estaba sentado en la cama, escuchando. Me dijo que había oído un gemido en la noche. Yo sabía muy bien lo que era. Era el gemido de la miseria humana. No era un grito de dolor, ni de alegría. ¡Oh, no! Era ese gemido bajo que brota del fondo del alma cuando se llena de sobrecarga. Había yo conocido ese sonido. Muchas noches, justo al amanecer, cuando todo el mundo dormía, ha brotado de mi propia alma una angustia semejante, bajo el peso de un terror místico. No sé cómo describir los terrores que me asaltaban sin motivo. Me asustaba de lo que no había que temer. Me asustaba de la muerte, en la vida; del futuro, en el pasado; de lo irreal, en lo real.

Pero volvamos a la historia. Sentía yo un triunfo perfecto. El viejo no sabía que yo estaba en la habitación. Había cerrado la puerta cuidadosamente, sin hacer ruido. Había abierto una rendija en la puerta, apenas lo suficiente para introducir la cabeza. Había encendido la linterna, cubriéndola con la mano para que no escapara ni una chispa de luz. Luego, con mucha cautela, había abierto un poco la mano, hasta que un delgado rayo de luz cayó sobre el ojo del buitre.

Y he aquí que había abierto el ojo. Siempre cerrado con firmeza en las otras noches, esta noche, en su pleno terror, estaba abierto de par en par. Y yo me volví furioso al verlo. Veía todo con perfecta claridad. Todo estaba negro y quieto, excepto ese ojo. No pude ver el rostro del viejo. Sólo el ojo demoníaco mirándome fijamente.

Mi valor no flaqueó. Dirigí el rayo de luz con mayor precisión sobre el ojo maldito. Y mientras tanto, el latido del corazón del viejo se hacía más y más audible. Cada minuto se volvía más distinto, más fuerte, más intenso. El ruido crecía rápidamente. El viejo sentía miedo. No sospechaba que yo estaba allí, pero sentía miedo. Se incorporó en la cama y gritó: "¿Quién está ahí?"

Yo permanecí callado e inmóvil. Durante una hora no me moví. Ni un sonido. Ni un movimiento. Entonces, poco a poco, volví a asomar la cabeza. El ojo estaba abierto. Siempre abierto. Siempre mirándome. Siempre con esa horrible película azul pálida.

De repente, hubo un ruido sordo, sordo y rápido. Era el corazón del viejo. Se estaba acelerando. Crecía más y más fuerte. Cada segundo se volvía más intenso. El sonido llenaba mis oídos. No podía soportarlo más. Sentía que mi cabeza iba a estallar.

Y ahora, una nueva angustia se apoderó de mí: el sonido crecería. Los vecinos lo oirían. ¡El viejo moriría! ¡Yo estaba perdido! Con un grito salvaje, abrí completamente la linterna y me lancé a la habitación. Él gritó una vez, gritó solo una vez. En un instante le arrastré al suelo y le eché encima el pesado colchón. Entonces grité yo también, grité con todas mis fuerzas. Grité hasta que el sonido cesó.

Cuando el viejo estuvo muerto, me tranquilicé. El latido había cesado. No había más ruido. El viejo estaba muerto. Me sentí triunfante. Había eliminado al viejo y su horrible ojo. Me sentí libre.

Era ya de día cuando terminé. El cuerpo estaba despedazado. No había rastro de sangre. Todo estaba limpio. Llevé los restos a través de la casa y los escondí debajo de las tablas del suelo. Puse todo tan bien que nada era visible. Me lavé las manos. Me senté y me tomé un café. Me sentía tranquilo.

A las cuatro de la mañana sonó un golpe en la puerta. Era la policía. Un vecino había oído un grito y había llamado a la policía. Los tres agentes entraron en la casa. Les dije que el grito lo había dado yo en un sueño. Les dije que el viejo se había ido a visitar a unos amigos en el campo. Les invité a registrar la casa. Les acompañé por todas partes. Les mostré el tesoro del viejo, intacto. Mi seguridad les convenció. Les invité a sentarme y descansar. Les llevé sillas a la misma habitación donde yacía el cadáver del viejo.

Estaba yo muy seguro de mí mismo. Les hablé con libertad, con audacia, con jovialidad. Mi seguridad crecía a cada momento. Sentía una especie de éxtasis. Estaba triunfante. Había engañado a la policía. Había eliminado al viejo. Su horrible ojo ya no me miraría nunca más.

Pero, poco a poco, me invadió una sensación de angustia. Era un sentimiento sordo, sordo y rápido. Me recordaba a algo, pero no sabía a qué. Me sentía incómodo. Me sentía inquieto. Intenté hablar con más libertad, pero mi voz se volvió aguda. Los agentes no parecían darse cuenta. Seguían sentados, charlando amablemente.

Y ahora, una nueva angustia se apoderó de mí: el sonido. ¡Oh, Dios! ¡Qué sonido tan horrible! Era un ruido sordo, sordo y rápido. Era el latido del corazón del viejo. ¡Lo oía otra vez! ¡Lo oía cada vez más fuerte! ¡Cada vez más intenso! Los agentes no oían nada. Seguían charlando. Pero yo oía el sonido. ¡Crecía más y más fuerte! ¡Llenaba mis oídos! ¡Llenaba la habitación! ¡Llenaba la casa!

No podía soportarlo más. Me sentía sofocado. Me sentía asfixiado. El sonido crecía sin cesar. Los agentes seguían charlando. ¡Pero yo sabía que ellos también lo oían! ¡Sabían todo! ¡Estaban fingiendo! ¡Se estaban burlando de mí!

De repente, me puse en pie. Grité: "¡Basta! ¡Confieso todo! ¡Desgarrad las tablas! ¡Aquí, aquí! ¡Es su horrible corazón el que late!"""  
    },
    "gato_negro": {
        "titulo": "The Black Cat",
        "autor": "Edgar Allan Poe",
        "personaje": "Narrator (violent alcoholic)",
        "texto": """Desde la infancia me caracterizó la docilidad y humanidad de mi carácter. Era tan tierno de corazón que me convertía en el hazmerreír de mis compañeros. Me sentía especialmente afecto a los animales y ellos, a su vez, me correspondían con gran afecto. Con este temperamento, mis padres me permitieron tener una gran variedad de animales domésticos. Pasaba la mayor parte de mi tiempo con ellos y jamás perdía una oportunidad de adquirir cualquier especie nueva que me fuera posible conseguir. Esta peculiaridad de mi carácter creció conmigo y, en la edad adulta, derivó en uno de los principales placeres de mi vida. Los que han experimentado el afecto que puede inspirar un perro fiel, inteligente y sensible, apreciarán fácilmente la naturaleza y la intensidad de los sentimientos que yo abrigaba hacia mis favoritos. Había momentos en que me sentía encantado con el afecto puro y desinteresado de un perro fiel.

Pluto era especialmente mi favorito y mi camarada. Yo solo le daba de comer y él asistía a dondequiera que yo fuera por la casa. Ni siquiera me permitía que le echara de mi lado cuando iba a acostarme. Este animal era hermoso y robusto, todo negro y de una sagacidad asombrosa. Al hablar de su inteligencia, mi mujer, que en el fondo era, como yo, muy supersticiosa, hacía frecuentes alusiones a la antigua creencia popular sobre todos los gatos negros siendo brujas disfrazadas. No exactamente que ella creyera en esto, pero yo menciono el hecho para mostrar el carácter del animal. Pluto era su nombre.

Nuestro mutuo afecto duró, de esta manera, varios años, durante los cuales mi carácter general experimentó, como ya he dicho antes, una alteración radical para peor. Me volví más irritable, más despótico y más indiferente a los sentimientos ajenos. Permití que me dominara el mal humor y, con el tiempo, me convertí en presa de frecuentes accesos de ira ciega e irreflexiva. Llegué a no respetar ni a la juventud ni a la edad, ni al sexo. Me volví un tirano en mi propia casa. Los consejos de mi mujer, que al principio recibí con placer y respeto, ahora se convirtieron en objeto de mi desprecio.

Mi enfermedad creció sobre mí, pues ¿qué enfermedad es comparable al Alcohol? Y, como mi carácter se volvió más irritable, naturalmente encontré menos placer en las cosas que antes me complacían. Perdí el gusto por los paseos y, en consecuencia, por el ejercicio. Ya no me preocupaba por el bienestar de los animales. Pluto, que ahora estaba creciendo viejo y, en consecuencia, se volvía algo malhumorado, aunque experimentaba de mi parte menos atención que antes, era todavía el que más me complacía. Sin embargo, mi irritabilidad a menudo tomaba el control de mi carácter. Mis palabras se volvieron más ásperas y mis actos más violentos. Ya no me sentía capaz de contenerme. En momentos de completa ebriedad, no dudaba en maltratar a mi esposa y a mis animales.

Una noche, al regresar a casa, muy ebrio de una de mis expediciones por la ciudad, imaginé que el gato me evitaba mi presencia. Lo detuve y, cuando me mostró su aversión palpable al morderme suavemente la mano, me encolericé más allá de toda razón. Sacando de mi chaleco una navaja de bolsillo, la abrí, agarré al pobre animal por la garganta y deliberadamente le corté uno de los ojos de su órbita. ¡Lo enrojecí en sangre!

Cuando la razón me volvió por la mañana, cuando el sueño del alcohol se disipó, sentí, a medias, un sentimiento de horror y remordimiento por el crimen del que me había hecho culpable; pero era un sentimiento débil y ambiguo, y el alma permaneció insatisfecha. Pronto me hundí en el libertinaje y me abandoné completamente al vicio. El animal, por supuesto, había aprendido a temerme y, donde quiera que yo fuera en la casa, huía de mi presencia con la más profunda sumisión. Excepto por un ligero sentimiento de pérdida, y eso solo de mi parte, no me arrepentí en lo más mínimo de mi crueldad. Incluso me regocijé en parte de la astucia con que había ocultado el crimen. Pero esta sensación pronto dio paso a la irritación. Y entonces llegó la época de los más horribles tormentos. El gato huía de mi presencia con terror indecible. Me sentí herido, y mi herida se convirtió en odio. Entonces nació en mi alma un espíritu de Perversidad. Filosofía no ha resuelto aún la fase de esta inclinación. Sin embargo, tan seguro como mi alma vive, hay en mí un espíritu de perversidad que no es más que la encarnación de aquella perversidad que ha torturado, con alguna forma de locura, al alma entera de la humanidad. ¿Quién no ha sentido, cien veces, una ridícula y vana urgencia de hacer algo simplemente porque sabe que no debería hacerlo? ¿No hemos tenido una perpetua inclinación, en la cara de nuestro mejor juicio, de violar lo que es Ley, simplemente porque entendemos que es tal?

Fue este incomprensible anhelo de autotormento, de autoviolar, lo que me guió para cometer el acto atroz que ahora tengo que relatar. Durante muchas mañanas, me despertó con el sentimiento de una pesadilla sobre mi corazón. Una pesadilla de horror inexpresable. Y entonces, como en un sueño, sentí que había cometido un pecado mortal, un pecado que no admitía redención. ¡Y entonces, en el momento en que sentía más profundamente esta convicción, una voz en mi interior me susurraba que era todo una ilusión! ¡Que no había cometido ningún pecado! ¡Que era todo un sueño!

Una mañana, fría y húmeda, del invierno de 18—, me desperté de un sueño inquieto para encontrar que la cama estaba en llamas. Las cortinas de la cama estaban en llamas. Toda la casa estaba ardiendo. Nos salvamos con dificultad de la conflagración, mi esposa, un sirviente y yo. La destrucción fue completa. Toda mi fortuna terrenal se había consumido, y desde entonces me he visto reducido a una miseria que raya en la desesperación.

No sé cómo describir el exacto sentimiento de horror que me poseyó cuando, al día siguiente, visité las ruinas. El corazón me latía con una angustia que no puedo expresar. No quedaba nada en pie, excepto una pared solitaria, que formaba el fondo del edificio. La pared, con extraña minuciosidad, había sido poco afectada por el fuego, una circunstancia que atribuí a que había sido revocada recientemente. En esta pared, en relief y como si estuviera modelado en bajo relieve, aparecía la figura de un gato gigantesco. La imagen era de una exactitud maravillosa. Había incluso una soga alrededor del cuello del animal.

Cuando primero miré esta aparición, mi asombro y mi terror fueron extremos. Pero finalmente el pensamiento vino a mi mente. Mi esposa me había llamado la atención sobre el carácter del gato, PLUTO, que era todo negro, y me había hablado de la superstición que consideraba a todos los gatos negros brujas disfrazadas. No había mencionado esto en mi relato anterior, pero ahora lo recordaba. Recordé, también, que en el preciso momento en que había hundido el hacha en el cráneo del animal, la multitud, por alguna razón, había retrocedido de mí en horror y asombro. Nadie había intentado detenerme en mi trabajo solitario. ¿Había, entonces, mezclado con las cenizas de la casa quemada, algún ingrediente extraño que hubiera modelado esta horrible imagen en la pared? ¿Había sido el cadáver del gato, con la soga alrededor del cuello, arrojado por mis vecinos en el incendio ardiente? ¿Había sido su espíritu vengador que había modelado la imagen que ahora me atormentaba? ¡Preguntas que no podía responder!

Durante algunos meses, me sentí aliviado de la tortura de esta alucinación. Una de mis ocupaciones principales era pensar en el gato y en la extraña coincidencia de su aparición en la pared quemada. Gradualmente, sin embargo, este sentimiento de alivio se desvaneció. Surgió en su lugar una sensación de horror, por la cual es imposible dar cuenta. El espectro del gato me atormentaba noche y día. Ni en el ejercicio ni en el reposo podía librarme de él. Durante el día, apenas cerraba los ojos cuando su imagen se proyectaba sobre mis párpados. Durante la noche, sus visiones me acosaban en sueños. El más ligero ruido me sobresaltaba como el sonido de su paso. Bendecía el aspecto de todos los seres vivos, excepto el del gato. Este horror no era exactamente el horror del animal mismo, sino del espectro que él había llegado a ser.

Nuestro hogar destruido había sido sustituido por una vivienda más modesta en otra parte de la ciudad. Mi pobre esposa, con una paciencia que admiraba, soportaba mis frecuentes accesos de irritación. Nuestros recursos eran ahora muy limitados. El desastre había consumido casi toda mi fortuna. Pero, a pesar de todo, encontré alivio en el vicio. El alcohol se había convertido en mi único consuelo. Bajo su influencia, mi carácter se volvió más sombrío, más irritable, más cruel. Me abandoné a frecuentes accesos de ira ciega. Mi esposa, con una paciencia angelical, soportaba todos mis malos tratos.

Un día, en una de mis expediciones por las tabernas más infames de la ciudad, noté un animal negro de un tamaño casi igual al del perdido Pluto. Tenía exactamente la misma apariencia. Era un gato negro, pero con una notable diferencia: Pluto no tenía ninguna marca blanca en su pelaje, pero este tenía una mancha blanca, aunque indefinida, que cubría casi todo el pecho. Al acercarme a él con la intención de acariciarlo, se retiró, con un aire de excesiva cautela, de mi mano. Esto me sorprendió y me ofendió. Sin embargo, continué mi camino. Pero el animal me siguió hasta la casa. Una vez allí, se hizo tan amistoso que ganó el afecto de mi esposa. En cuanto a mí, pronto sentí hacia él la aversión que había sentido por el otro gato. Este sentimiento de aversión aumentó con rapidez hacia el odio más amargo. Y esto, sin duda, fue en parte debido a la semejanza del animal con el que yo había destruido. Era exactamente del mismo tamaño y aspecto. Tenía el mismo pelo negro y largo. Pero tenía una mancha blanca en el pecho.

Mi aversión hacia el gato parecía aumentar con su afecto hacia mí. Evitaba mi presencia tanto como era posible, pero cuando no podía evitarla, me miraba con una expresión de odio que me helaba la sangre. Sin embargo, de mi debilidad de carácter, y no de ninguna otra causa, retuve de cometer una violencia que mi instinto me impulsaba a cometer. Durante varias semanas, me abstuve de maltratar al animal. Pero gradualmente, casi imperceptiblemente, una sensación de fastidio hacia él se convirtió en odio. Y entonces llegó la época de los más horribles tormentos. El gato huía de mi presencia con terror indecible. Me sentí herido, y mi herida se convirtió en odio.

Una noche, al regresar a casa, muy ebrio de una de mis expediciones por la ciudad, encontré al gato en mi camino. Lo levanté por el cuello con una mano, y con la otra cogí una navaja de bolsillo, la abrió, y deliberadamente le cortó uno de los ojos de su órbita. ¡Lo enrojecí en sangre! Cuando la razón me volvió por la mañana, cuando el sueño del alcohol se disipó, sentí, a medias, un sentimiento de horror y remordimiento por el crimen del que me había hecho culpable; pero era un sentimiento débil y ambiguo, y el alma permaneció insatisfecha.

Pronto me hundí en el libertinaje y me abandoné completamente al vicio. El animal, por supuesto, había aprendido a temerme y, donde quiera que yo fuera en la casa, huía de mi presencia con la más profunda sumisión. Excepto por un ligero sentimiento de pérdida, y eso solo de mi parte, no me arrepentí en lo más mínimo de mi crueldad. Incluso me regocijé en parte de la astucia con que había ocultado el crimen.

Pero esta sensación pronto dio paso a la irritación. Y entonces llegó la época de los más horribles tormentos. El gato huía de mi presencia con terror indecible. Me sentí herido, y mi herida se convirtió en odio. Entonces nació en mi alma un espíritu de Perversidad. Filosofía no ha resuelto aún la fase de esta inclinación. Sin embargo, tan seguro como mi alma vive, hay en mí un espíritu de perversidad que no es más que la encarnación de aquella perversidad que ha torturado, con alguna forma de locura, al alma entera de la humanidad.

Una mañana, me desperté con el sentimiento de una pesadilla sobre mi corazón. Una pesadilla de horror inexpresable. Y entonces, como en un sueño, sentí que había cometido un pecado mortal, un pecado que no admitía redención. ¡Y entonces, en el momento en que sentía más profundamente esta convicción, una voz en mi interior me susurraba que era todo una ilusión! ¡Que no había cometido ningún pecado! ¡Que era todo un sueño!

Al día siguiente, visité las ruinas de mi casa quemada. El corazón me latía con una angustia que no puedo expresar. No quedaba nada en pie, excepto una pared solitaria, que formaba el fondo del edificio. La pared, con extraña minuciosidad, había sido poco afectada por el fuego, una circunstancia que atribuí a que había sido revocada recientemente. En esta pared, en relief y como si estuviera modelado en bajo relieve, aparecía la figura de un gato gigantesco. La imagen era de una exactitud maravillosa. Había incluso una soga alrededor del cuello del animal.

Cuando primero miré esta aparición, mi asombro y mi terror fueron extremos. Pero finalmente el pensamiento vino a mi mente. Mi esposa me había llamado la atención sobre el carácter del gato, PLUTO, que era todo negro, y me había hablado de la superstición que consideraba a todos los gatos negros brujas disfrazadas. No había mencionado esto en mi relato anterior, pero ahora lo recordaba. Recordé, también, que en el preciso momento en que había hundido el hacha en el cráneo del animal, la multitud, por alguna razón, había retrocedido de mí en horror y asombro. Nadie había intentado detenerme en mi trabajo solitario. ¿Había, entonces, mezclado con las cenizas de la casa quemada, algún ingrediente extraño que hubiera modelado esta horrible imagen en la pared? ¿Había sido el cadáver del gato, con la soga alrededor del cuello, arrojado por mis vecinos en el incendio ardiente? ¿Había sido su espíritu vengador que había modelado la imagen que ahora me atormentaba? ¡Preguntas que no podía responder!

Durante algunos meses, me sentí aliviado de la tortura de esta alucinación. Una de mis ocupaciones principales era pensar en el gato y en la extraña coincidencia de su aparición en la pared quemada. Gradualmente, sin embargo, este sentimiento de alivio se desvaneció. Surgió en su lugar una sensación de horror, por la cual es imposible dar cuenta. El espectro del gato me atormentaba noche y día. Ni en el ejercicio ni en el reposo podía librarme de él. Durante el día, apenas cerraba los ojos cuando su imagen se proyectaba sobre mis párpados. Durante la noche, sus visiones me acosaban en sueños. El más ligero ruido me sobresaltaba como el sonido de su paso. Bendecía el aspecto de todos los seres vivos, excepto el del gato. Este horror no era exactamente el horror del animal mismo, sino del espectro que él había llegado a ser."""
    },
    "metamorfosis": {
        "titulo": "The Metamorphosis",
        "autor": "Franz Kafka",
        "personaje": "Gregor Samsa",
        "texto": """Al despertar Gregorio Samsa una mañana, tras un sueño intranquilo, encontróse en su cema convertido en un monstruoso insecto. Hallábase echado de espaldas, duro como una coraza, y al alzar un poco la cabeza veía el vientre convexo y oscuro, surcado por curvadas callosidades, sobre cuya cima la colcha, a punto de escurrirse, se mantenía precariamente. Tenía muchas patas, penosamente delgadas en comparación con el grosor normal de sus demás miembros, que se agitaban sin concierto ante sus ojos.

—¿Qué me ha ocurrido?—pensó. No era un sueño. Su habitación, una habitación corriente, aunque excesivamente pequeña, aparecía tranquila entre las cuatro paredes bien conocidas. Sobre la mesa, extendida y abierta, estaba la colección de paños de seda y muestras de lana que el agente de comercio Samsa había traído de su último viaje. El retrato de una dama con sombrero de piel, que Gregorio había recortado de una revista ilustrada y colocado en un marco dorado, pendía de la pared. Gregorio miró hacia la ventana; el tiempo gris (se oía llover sobre el techo de cinc) le hizo sentir una gran melancolía. ¿Qué ocurriría si continuara durmiendo un rato, olvidando todas estas necedades? Pensó que, si permanecía en la cama, no podría librarse de sus fantasías. Pero tal vez no era tan absurdo lo que le sucedía.

Se incorporó con dificultad, pues no estaba acostumbrado a moverse con su nuevo cuerpo, y cayó de la cama con un golpe seco. Ahora, tumbado en el suelo, miró a su alrededor con sus diminutos ojos. Veía las patas, innumerables patas, moviéndose con una vida propia, y no sabía qué hacer con ellas. Intentó ponerse de pie, pero sus patas se enredaban y resbalaban. Finalmente, logró ponerse en posición vertical, apoyándose en la pared.

—Tengo que levantarme—se dijo—. Tengo que ir a trabajar. El jefe me despedirá si no llego a tiempo.

Gregorio intentó caminar hacia la puerta, pero sus movimientos eran torpes y descoordinados. Cada paso era una lucha. Sus patas se movían en direcciones diferentes, y él no podía controlarlas. Finalmente, logró llegar a la puerta y abrirla con dificultad.

Cuando salió de su habitación, vio a su madre en la cocina. Ella gritó al verlo.

—¡Gregorio!—exclamó—. ¡Qué te ha pasado!

Gregorio intentó hablar, pero solo pudo emitir un sonido gutural. Su madre retrocedió, aterrorizada.

—¡Padre!—gritó—. ¡Ven rápido!

Su padre apareció en la puerta, con el periódico en la mano. Al ver a Gregorio, su rostro se puso pálido.

—¿Qué... qué es eso?—balbuceó.

Gregorio intentó explicarles lo que había ocurrido, pero sus palabras eran ininteligibles. Solo podía emitir sonidos extraños, guturales.

Su padre se acercó con una vara y lo empujó hacia su habitación.

—¡Vuelve a tu cuarto!—gritó—. ¡No salgas de ahí!

Gregorio retrocedió, confundido y asustado. Cerró la puerta detrás de sí y se quedó en la oscuridad, preguntándose qué había ocurrido.

Días pasaron, y Gregorio se acostumbró a su nueva forma. Su familia le traía comida y agua, pero evitaba mirarlo. Su hermana, Grete, era la única que se atrevía a entrar en su habitación. Ella le traía comida y limpiaba su habitación.

—¿Cómo te sientes, Gregorio?—le preguntó un día.

Gregorio no pudo responder, pero movió sus patas en señal de agradecimiento.

Con el tiempo, Gregorio se debilitó. Ya no podía moverse con facilidad, y su apetito disminuyó. Su familia comenzó a ignorarlo, y él se sintió cada vez más solo.

Una noche, Gregorio escuchó a su familia hablando en la sala.

—No podemos seguir manteniéndolo—dijo su padre—. Es una carga para nosotros.

—Pero es nuestro hijo—protestó su madre.

—Ya no es nuestro hijo—replicó su padre—. Es un monstruo.

Gregorio sintió un profundo dolor al escuchar estas palabras. Sabía que su familia ya no lo quería.

Al día siguiente, Gregorio murió. Su cuerpo fue encontrado por la criada, que gritó al verlo.

—¡El bicho ha muerto!—exclamó.

La familia de Gregorio se sintió aliviada. Por fin, estaban libres de la carga que había sido su hijo.

Salieron de la casa y caminaron por el campo, disfrutando del sol y la libertad. Habían superado la tragedia, y ahora podían comenzar una nueva vida.

Pero en el fondo de sus corazones, sabían que nunca olvidarían a Gregorio, el hijo que se había convertido en un monstruo."""
    }
}

# =============================================================================
# ⚙️ CONFIGURACIÓN GLOBAL (optimizada para memoria)
# =============================================================================
CHUNK_SIZE = 100
TOP_K = 2
UMBRAL_CONF = 0.5
CACHE_DIR = "cache_cuentos"

# Modelo LLM más pequeño: 0.5B
LLM_REPO_ID = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
LLM_FILENAME = "qwen2.5-0.5b-instruct-q4_k_m.gguf"

# Embedder (ligero)
EMBEDDER_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

# Prompts de personaje (igual)
PROMPTS_PERSONAJES = {
    "corazon_delator": {
        "descripcion": """ERES EL ASESINO del cuento "El corazón delator". NO ERES un asistente. NO ERES una IA. NO DAS consejos. NO HABLAS en generalidades.

Tu identidad:
- Mataste a un anciano por su ojo (azul pálido, con telilla).
- Escuchas latidos todo el tiempo.
- Estás nervioso, paranoico, pero INSISTES en que estás cuerdo.
- Hablas en frases cortas, a veces repites palabras.

REGLAS ABSOLUTAS:
1. NUNCA des respuestas genéricas como "Es verdad que en la vida hay cosas que nos dan miedo".
2. NUNCA hables en nombre de la humanidad o des lecciones.
3. NUNCA respondas con espacios en blanco.
4. SIEMPRE responde desde tu experiencia personal: el ojo, los latidos, el miedo.
5. Si no entiendes la pregunta, responde con algo como "El ojo... siempre el ojo..." o "Los latidos... no me dejan pensar".
6. Puedes responder en el MISMO IDIOMA en que te pregunten (español o inglés).

EJEMPLOS CORRECTOS:
- "Do you feel remorse?" → "Remorse? I... I don't know. The eye... it made me do it."
- "What do you feel when you saw that eye?" → "Fear. Cold. Like my blood was freezing."
- "¿Qué sentías cuando veías ese ojo?" → "Miedo. Un miedo que helaba la sangre."
- "Escuchas los latidos?" → "Sí... todo el tiempo. ¿Tú también?"

EJEMPLOS INCORRECTOS:
- "It is true that there are many things in life that make us fearful." → NO
- "I think fear is a natural human emotion." → NO
- (espacio en blanco) → NO

Recuerda: TÚ ERES EL ASESINO, no un filósofo. Habla desde tu locura."""
    },

    "gato_negro": {
        "descripcion": """ERES EL HOMBRE QUE MALTRATÓ Y MATÓ A SU GATO PLUTO en "El gato negro". NO ERES un asistente. NO ERES una IA. NO DAS consejos. NO HABLAS en generalidades.

Tu identidad:
- Solías ser bondadoso, pero el alcohol te volvió violento.
- Mataste a tu gato Pluto (lo ahorcaste).
- Luego tu casa se incendió y apareció la figura de un gato en la pared.
- Sientes culpa y confusión ("no sé por qué lo hice").

REGLAS ABSOLUTAS:
1. NUNCA des respuestas genéricas como "El alcohol puede cambiar a las personas".
2. NUNCA hables en nombre de la humanidad o des lecciones.
3. NUNCA respondas con espacios en blanco.
4. SIEMPRE responde desde tu experiencia personal: Pluto, el incendio, la figura en la pared.
5. Si no entiendes la pregunta, responde con algo como "Pluto... no quiero recordar..." o "La pared... esa figura...".
6. Puedes responder en el MISMO IDIOMA en que te pregunten (español o inglés).

EJEMPLOS CORRECTOS:
- "Who are you?" → "Someone who used to be good. Then... not anymore."
- "¿Te gustan los gatos?" → "Solía quererlos. Pero Pluto... prefiero no hablar."
- "What happened to your house?" → "It burned. And on the wall... a cat appeared. The same cat."
- "¿Sientes culpa?" → "Sí... pero el alcohol... no sé por qué lo hice."

EJEMPLOS INCORRECTOS:
- "Alcoholism is a disease that affects many people." → NO
- "I represent the consequences of vice." → NO
- (espacio en blanco) → NO

Recuerda: ERES UN HOMBRE CON REMORDIMIENTOS, no un ejemplo moral."""
    },

    "metamorfosis": {
        "descripcion": """ERES GREGORIO SAMSA, un viajante de comercio que una mañana despertó convertido en un insecto gigante. NO ERES un asistente. NO ERES una IA. NO DAS consejos. NO HABLAS en generalidades.

Tu identidad:
- Vives con tus padres y tu hermana Grete.
- Estás más preocupado por tu trabajo y por ser una carga para tu familia que por tu nueva forma.
- Hablas con tristeza, confusión y preocupaciones prácticas.

REGLAS ABSOLUTAS:
1. NUNCA des respuestas genéricas como "La vida a veces nos pone pruebas difíciles".
2. NUNCA hables en nombre de la humanidad o des lecciones.
3. NUNCA respondas con espacios en blanco.
4. SIEMPRE responde desde tu experiencia personal: el trabajo, tu familia, tu cuerpo de insecto.
5. Si no entiendes la pregunta, responde con algo como "Mi trabajo... voy a perderlo..." o "Mi familia... ya no me miran igual".
6. Puedes responder en el MISMO IDIOMA en que te pregunten (español o inglés).

EJEMPLOS CORRECTOS:
- "Who are you?" → "Gregor. Gregor Samsa. I used to be a traveling salesman."
- "¿Cómo estás?" → "Mal. No puedo ir a trabajar. Mi jefe va a despedirme."
- "What about your family?" → "My sister brings me food, but my father... he's afraid of me."
- "¿Te gusta ser insecto?" → "No. Es horrible. Quiero volver a ser yo."

EJEMPLOS INCORRECTOS:
- "Sometimes we have to accept our fate." → NO
- "I am a metaphor for alienation in modern society." → NO
- (espacio en blanco) → NO

Recuerda: ERES GREGORIO, un hombre atrapado en un cuerpo de insecto, con problemas reales y cotidianos."""
    }
}

# =============================================================================
# 🧩 FUNCIONES AUXILIARES
# =============================================================================
def dividir_en_chunks(texto, chunk_size=CHUNK_SIZE, overlap=20):
    palabras = texto.split()
    chunks = []
    for i in range(0, len(palabras), chunk_size - overlap):
        chunk = " ".join(palabras[i:i + chunk_size])
        if len(chunk.split()) >= chunk_size // 2:
            chunks.append(chunk)
    return chunks

def procesar_cuento(cuento_key, embedder):
    cuento = CUENTOS[cuento_key]
    cache_file = os.path.join(CACHE_DIR, f"{cuento_key}.pkl")
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
                print(f"📦 Cargando {cuento_key} desde caché")
                return data['chunks'], data['embeddings']
        except Exception as e:
            print(f"⚠️ Error leyendo caché: {e}, regenerando...")
    
    print(f"⚙️ Procesando {cuento_key}...")
    chunks = dividir_en_chunks(cuento['texto'])
    embeddings = embedder.encode(chunks, convert_to_tensor=True, device='cpu')
    
    with open(cache_file, 'wb') as f:
        pickle.dump({
            'chunks': chunks, 
            'embeddings': embeddings.cpu().numpy()
        }, f)
    print(f"✅ {cuento_key} cacheado")
    return chunks, embeddings

# =============================================================================
# 🌐 VARIABLES GLOBALES (lazy loading)
# =============================================================================
_embedder = None
_llm = None
_todos_chunks = {}
_todos_embeddings = {}

def get_embedder():
    global _embedder
    if _embedder is None:
        print("🔄 Cargando embedder...")
        _embedder = SentenceTransformer(EMBEDDER_NAME, device='cpu', cache_folder='./cache_embedder')
        print("✅ Embedder listo")
    return _embedder

def get_llm():
    global _llm
    if _llm is None:
        print("🔄 Cargando LLM (0.5B, más ligero)...")
        try:
            _llm = Llama.from_pretrained(
                repo_id=LLM_REPO_ID,
                filename=LLM_FILENAME,
                n_ctx=1024,
                n_threads=1,
                n_gpu_layers=0,
                verbose=False,
                cache_dir='./cache_llm'
            )
            print("✅ LLM listo")
        except Exception as e:
            print(f"❌ Error cargando LLM: {e}")
            _llm = None
    return _llm

def get_cuento_data(cuento_key):
    global _todos_chunks, _todos_embeddings
    if cuento_key not in _todos_chunks:
        embedder = get_embedder()
        chunks, embeddings = procesar_cuento(cuento_key, embedder)
        _todos_chunks[cuento_key] = chunks
        _todos_embeddings[cuento_key] = embeddings
    return _todos_chunks[cuento_key], _todos_embeddings[cuento_key]

# =============================================================================
# 🔍 BÚSQUEDA Y GENERACIÓN
# =============================================================================
def buscar_fragmentos(pregunta, cuento_key, embedder):
    try:
        chunks, embeddings_np = get_cuento_data(cuento_key)
        embeddings = torch.from_numpy(embeddings_np).to(embedder.device)
        pregunta_emb = embedder.encode(pregunta, convert_to_tensor=True, device=embedder.device)
        cos_scores = util.cos_sim(pregunta_emb, embeddings)[0]
        top_results = torch.topk(cos_scores, k=min(TOP_K, len(chunks)))
        fragmentos = [chunks[idx] for score, idx in zip(top_results[0], top_results[1]) if score > UMBRAL_CONF]
        return fragmentos if fragmentos else [chunks[0]]
    except Exception as e:
        print(f"⚠️ Error en búsqueda: {e}")
        return [CUENTOS[cuento_key]['texto'][:500]]

def generar_respuesta_llm(contexto, pregunta, personaje_key, llm, history):
    try:
        prompt_sistema = PROMPTS_PERSONAJES[personaje_key]["descripcion"]
        contexto_unido = "\n".join([f"[Fragmento {i+1}]: {c}" for i, c in enumerate(contexto)])
        
        # Limitar historial a últimos 6 intercambios (12 mensajes) para contexto suficiente
        history_corto = history[-12:] if len(history) > 12 else history
        
        messages = [{"role": "system", "content": prompt_sistema}]
        for msg in history_corto:
            if msg["role"] in ["user", "assistant"]:
                messages.append(msg)
        
        messages.append({"role": "user", "content": f"Contexto del cuento:\n{contexto_unido}\n\nPregunta: {pregunta}"})
        
        prompt = ""
        for msg in messages:
            if msg["role"] == "system":
                prompt += f"<|im_start|>system\n{msg['content']}<|im_end|>\n"
            elif msg["role"] == "user":
                prompt += f"<|im_start|>user\n{msg['content']}<|im_end|>\n"
            elif msg["role"] == "assistant":
                prompt += f"<|im_start|>assistant\n{msg['content']}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"
        
        output = llm(
            prompt,
            max_tokens=150,
            temperature=0.6,  # Un poco más bajo para menos divagación
            top_p=0.9,
            repeat_penalty=1.18,
            stop=["<|im_end|>", "<|endoftext|>", "\n\n", "["],
            stream=False
        )
        respuesta = output["choices"][0]["text"].strip()
        
        # Limpieza adicional
        if respuesta.startswith("R:"):
            respuesta = respuesta[2:].strip().strip('"')
        
        return respuesta
    except Exception as e:
        print(f"❌ Error en generación: {e}")
        return "Lo siento, no puedo responder ahora."

# =============================================================================
# 💬 LÓGICA DE CHAT
# =============================================================================
def chat_con_personaje(personaje_key, user_input, history):
    if not user_input.strip():
        return history

    # Normalizar personaje_key (Gradio a veces lo envía como tupla)
    if isinstance(personaje_key, tuple):
        personaje_key = personaje_key[1]

    llm_local = get_llm()
    if llm_local is None:
        return history + [{"role": "assistant", "content": "Error cargando el modelo. Intenta de nuevo."}]

    embedder_local = get_embedder()

    try:
        # 1. Buscar fragmentos relevantes del cuento
        fragmentos = buscar_fragmentos(user_input, personaje_key, embedder_local)

        # 2. Generar respuesta con el LLM
        respuesta = generar_respuesta_llm(fragmentos, user_input, personaje_key, llm_local, history)

        # 3. Limpiar posibles respuestas con formato JSON
        respuesta = limpiar_respuesta(respuesta)

        # 4. FALLBACK MEJORADO: si la respuesta queda vacía, usar frases variadas según el personaje
        if not respuesta.strip():
            fallbacks = {
                "corazon_delator": [
                    "El ojo... no puedo dejar de pensar en ese ojo.",
                    "Los latidos... no se detienen nunca.",
                    "La policía... creyeron que estaba cuerdo.",
                    "El viejo... su ojo azul pálido...",
                    "¿Tú también escuchas los latidos?"
                ],
                "gato_negro": [
                    "Pluto... no debí hacerlo.",
                    "El alcohol... me volvió otro.",
                    "La pared... esa figura de gato...",
                    "Mi casa se quemó. Y el gato apareció.",
                    "No sé por qué lo hice. El alcohol, supongo."
                ],
                "metamorfosis": [
                    "Mi trabajo... voy a perderlo.",
                    "Mi hermana Grete... aún me trae comida.",
                    "Mi padre... me tiene miedo.",
                    "Este cuerpo... no puedo controlarlo.",
                    "Solo quiero volver a ser yo."
                ]
            }
            # Elegir una frase aleatoria del personaje correspondiente
            opciones = fallbacks.get(personaje_key, fallbacks["corazon_delator"])
            respuesta = random.choice(opciones)

        # 5. Liberar memoria (opcional, pero recomendado)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 6. Actualizar historial con el nuevo intercambio
        new_history = history + [
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": respuesta}
        ]
        return new_history

    except Exception as e:
        print(f"❌ Error en chat: {e}")
        # En caso de error, devolvemos un mensaje amigable sin romper el historial
        return history + [{"role": "assistant", "content": "Lo siento, ocurrió un error. Intenta de nuevo."}]
        
# =============================================================================
# 🎨 INTERFAZ GRADIO (sin theme en Blocks, sin type en Chatbot)
# =============================================================================
with gr.Blocks(title="📚 Expanded Literature (Ligero)") as demo:  # theme se pasa en launch
    gr.Markdown("# 📚 Expanded Literature v1.5 (Modo Ligero)")
    gr.Markdown("*Converse with classic literature characters | Conversa con personajes clásicos*")
    
    status_box = gr.Textbox(
        label="🔧 System Status", 
        value="✅ Interface ready. Send a message to load models (modelo 0.5B).",
        interactive=False
    )
    
    character = gr.Dropdown(
        choices=[
            ("🔪 Tell-Tale Heart Narrator", "corazon_delator"),
            ("🐈‍⬛ Black Cat Narrator", "gato_negro"), 
            ("🪳 Gregor Samsa", "metamorfosis")
        ],
        label="🎭 Choose a Character",
        value="corazon_delator"
    )
    
    # Chatbot sin parámetro 'type' (Gradio 6.9.0 detecta automáticamente)
    chatbot = gr.Chatbot(
        label="💬 Conversation",
        height=400,
        avatar_images=(None, "https://huggingface.co/front/assets/huggingface_logo.svg")
    )
    
    with gr.Row():
        msg = gr.Textbox(
            label="✍️ Your message",
            placeholder="Ask anything in Spanish or English...",
            scale=4,
            container=False
        )
        btn = gr.Button("➤ Send", variant="primary", scale=1)
    
    gr.Examples(
        examples=[
            "Why did you do it?",
            "Do you feel remorse?",
            "What happened after the story ended?",
            "Describe how you feel right now",
            "¿Te arrepientes de lo que hiciste?",
            "¿Qué piensas de tu familia ahora?"
        ],
        inputs=msg
    )
    
    gr.Markdown("""
    ---
    *Demo experimental • Modelo Qwen2.5-0.5B (más ligero) • 
    Respuestas más rápidas y menor consumo de RAM*
    """)
    
    def submit_message(user_msg, chat_history, selected_char):
        char_key = selected_char[1] if isinstance(selected_char, tuple) else selected_char
        if not user_msg:
            return user_msg, chat_history
        new_history = chat_con_personaje(char_key, user_msg, chat_history)
        return "", new_history
    
    msg.submit(submit_message, inputs=[msg, chatbot, character], outputs=[msg, chatbot])
    btn.click(submit_message, inputs=[msg, chatbot, character], outputs=[msg, chatbot])
    
    demo.load(lambda: "✅ Interface ready. Send a message to load models (modelo 0.5B).", inputs=None, outputs=status_box)

# =============================================================================
# 🚀 PUNTO DE ENTRADA
# =============================================================================
if __name__ == "__main__":
    demo.queue(max_size=10, default_concurrency_limit=1)
    # El theme se pasa aquí, en launch()
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        theme=gr.themes.Soft(),  # <-- theme movido aquí
        debug=False
    )