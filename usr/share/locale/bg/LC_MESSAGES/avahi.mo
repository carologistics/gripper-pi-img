��    �        �   
      �  �  �  h   1  �   �  i  �  b  �  �   X     �       %     5   :     p     ~     �     �  "   �     �      �     �       	   /     9  "   N  4   q  *   �  .   �        
             %     7     K     ]     z     �     �     �     �     �     �          %     ;     Q     f          �     �     �     �     �     �               9     T  %   t  &   �  #   �  #   �  #   	  !   -  (   O  <   x  %   �     �      �       #   9     ]     }  #   �  %   �  ?   �  	        )  %   =     c  
   s     ~     �     �     �     �     �     �     �          !     5     C     U  4   m     �     �     �     �     �               -     E     Z     c     x     �     �  '   �     �  &   �  	   �                 #      &      /      @      E      U   L   o   ;   �      �   "   !     :!     G!     U!     b!     p!     t!  	   }!     �!  *   �!  $   �!  +   �!  #   "  7   0"  %   h"  "   �"  4   �"  (   �"  (   #     8#     H#     [#     o#     �#     �#     �#     �#     �#  
   �#  &   �#  '   $  ,   *$     W$     ]$     r$     v$      �$  �  �$  %  L&  �   r*  �  %+  <  �,  �  �.    �2     �3  !   �3  +   �3  s   '4  "   �4  
   �4     �4     �4  ,   �4  /   5  C   N5  7   �5  1   �5     �5  1   6  E   J6  ]   �6  A   �6  M   07     ~7     �7     �7  %   �7  .   �7  %   8  /   98  &   i8     �8     �8     �8     �8     9     +9     J9     i9     �9     �9  /   �9  )   �9     ":  =   A:     :     �:  1   �:  &   �:  *   �:  8   ;  :   W;  C   �;  O   �;  j   &<  ]   �<  P   �<  g   @=  G   �=  j   �=  �   [>  a   �>  D   E?  N   �?  P   �?  N   *@  I   y@  0   �@  C   �@  C   8A  _   |A     �A  7   �A  ?   -B     mB     �B  8   �B  !   �B  1   �B     .C  -   NC  %   |C     �C  %   �C  '   �C  -   D  #   >D  -   bD  9   �D  Y   �D  %   $E     JE  -   jE  2   �E     �E  -   �E  -   F  7   GF  (   F     �F  3   �F     �F  #   	G     -G  S   4G  '   �G  =   �G     �G     H     "H  
   =H  ;   HH  #   �H     �H  $   �H  L   �H  �   #I  �   �I  >   IJ  G   �J     �J     �J     K     K     4K     8K     QK     kK  9   �K  =   �K  U    L  3   VL  u   �L  E    M  7   FM  Z   ~M  +   �M  /   N     5N  1   QN  1   �N  ,   �N  *   �N  *   O     8O  &   ?O  (   fO     �O  1   �O  2   �O  7   P     =P  :   FP     �P  *   �P  +   �P     C   �   ,       '                                      r      �       �   �   T   L   |       \   %   9      e   "   �   �   H       A   �   �       �   �       �          }   j          h         +   �           #   (               �      �   �                 k   [   <   �   �       D   0   ]   x   t   w   Q   s           �   z   7   O       a   d          .           Y       {   R       5   K   y   )       /       U       c      �       !       �   �   
   l   ;      v              �          3   �   p   _       n         u      E       S   W   8   M   N   ^       1      I   *   F   o           b   �   @   g              �       �   q           2   6   V          X   �      ?           &       P   �          i   	             -      =       $       �           m   �   �   J          G      �      >   f   ~           4   B           :   Z   `        -h --help            Show this help
    -V --version         Show version
    -D --browse-domains  Browse for browsing domains instead of services
    -a --all             Show all services, regardless of the type
    -d --domain=DOMAIN   The domain to browse in
    -v --verbose         Enable verbose mode
    -t --terminate       Terminate after dumping a more or less complete list
    -c --cache           Terminate after dumping all entries from the cache
    -l --ignore-local    Ignore local services
    -r --resolve         Resolve services found
    -f --no-fail         Don't fail if the daemon is not available
    -p --parsable        Output in parsable format
     -k --no-db-lookup    Don't lookup service types
    -b --dump-db         Dump service type database
 %s [options]

    -h --help            Show this help
    -s --ssh             Browse SSH servers
    -v --vnc             Browse VNC servers
    -S --shell           Browse both SSH and VNC
    -d --domain=DOMAIN   The domain to browse in
 %s [options] %s <host name ...>
%s [options] %s <address ... >

    -h --help            Show this help
    -V --version         Show version
    -n --name            Resolve host name
    -a --address         Resolve address
    -v --verbose         Enable verbose mode
    -6                   Lookup IPv6 address
    -4                   Lookup IPv4 address
 %s [options] %s <name> <type> <port> [<txt ...>]
%s [options] %s <host-name> <address>

    -h --help            Show this help
    -V --version         Show version
    -s --service         Publish service
    -a --address         Publish address
    -v --verbose         Enable verbose mode
    -d --domain=DOMAIN   Domain to publish service in
    -H --host=DOMAIN     Host where service resides
       --subtype=SUBTYPE An additional subtype to register this service with
    -R --no-reverse      Do not publish reverse entry with address
    -f --no-fail         Don't fail if the daemon is not available
 %s [options] <new host name>

    -h --help            Show this help
    -V --version         Show version
    -v --verbose         Enable verbose mode
 : All for now
 : Cache exhausted
 <i>No service currently selected.</i> A NULL terminated list of service types to browse for Access denied Address Address family Address: An unexpected D-Bus error occurred Avahi client failure: %s Avahi domain browser failure: %s Avahi resolver failure: %s Bad number of arguments
 Bad state Browse Service Types Browse service type list is empty! Browsing for service type %s in domain %s failed: %s Browsing for services in domain <b>%s</b>: Browsing for services on <b>local network</b>: Browsing... Canceled.
 Change domain Choose SSH server Choose Shell Server Choose VNC server Client failure, exiting: %s
 Connecting to '%s' ...
 DNS failure: FORMERR DNS failure: NOTAUTH DNS failure: NOTIMP DNS failure: NOTZONE DNS failure: NXDOMAIN DNS failure: NXRRSET DNS failure: REFUSED DNS failure: SERVFAIL DNS failure: YXDOMAIN DNS failure: YXRRSET Daemon connection failed Daemon not running Desktop Disconnected, reconnecting ...
 Domain Domain Name: E Ifce Prot %-*s %-20s Domain
 E Ifce Prot Domain
 Established under name '%s'
 Failed to add address: %s
 Failed to add service: %s
 Failed to add subtype '%s': %s
 Failed to connect to Avahi server: %s Failed to create address resolver: %s
 Failed to create browser for %s: %s Failed to create client object: %s
 Failed to create domain browser: %s Failed to create entry group: %s
 Failed to create host name resolver: %s
 Failed to create resolver for %s of type %s in domain %s: %s Failed to create simple poll object.
 Failed to parse address '%s'
 Failed to parse port number: %s
 Failed to query host name: %s
 Failed to query version string: %s
 Failed to read Avahi domain: %s Failed to register: %s
 Failed to resolve address '%s': %s
 Failed to resolve host name '%s': %s
 Failed to resolve service '%s' of type '%s' in domain '%s': %s
 Host Name Host name conflict
 Host name successfully changed to %s
 Initializing... Interface: Invalid DNS TTL Invalid DNS class Invalid DNS return code Invalid DNS type Invalid Error Code Invalid RDATA Invalid address Invalid argument Invalid configuration Invalid domain name Invalid flags Invalid host name Invalid interface index Invalid number of arguments, expecting exactly one.
 Invalid operation Invalid packet Invalid port number Invalid protocol specification Invalid record Invalid record key Invalid service name Invalid service subtype Invalid service type Is empty Local name collision Location Memory exhausted Name Name collision, picking new name '%s'.
 No command specified.
 No suitable network protocol available Not found Not permitted Not supported OK OS Error Operation failed Port Resolve Service Resolve Service Host Name Resolve the host name of the selected service automatically before returning Resolve the selected service automatically before returning Resource record key is pattern Server version: %s; Host name: %s
 Service Name Service Name: Service Type Service Type: TXT TXT Data TXT Data: Terminal The IP port number of the resolved service The TXT data of the resolved service The address family for host name resolution The address of the resolved service The domain to browse in, or NULL for the default domain The host name of the resolved service The object passed in was not valid The requested operation is invalid because redundant The service name of the selected service The service type of the selected service Timeout reached Too few arguments
 Too many arguments
 Too many clients Too many entries Too many objects Type Version mismatch Waiting for daemon ...
 _Domain... avahi_domain_browser_new() failed: %s
 avahi_service_browser_new() failed: %s
 avahi_service_type_browser_new() failed: %s
 empty execlp() failed: %s
 n/a service_browser failed: %s
 service_type_browser failed: %s
 Project-Id-Version: Avahi
Report-Msgid-Bugs-To: https://github.com/lathiat/avahi/issues
PO-Revision-Date: 2013-11-20 09:58+0000
Last-Translator: Alexander Shopov <ash@kambanaria.org>
Language-Team: Bulgarian (http://www.transifex.com/lennart/avahi/language/bg/)
Language: bg
MIME-Version: 1.0
Content-Type: text/plain; charset=UTF-8
Content-Transfer-Encoding: 8bit
Plural-Forms: nplurals=2; plural=(n != 1);
     -h --help            Извеждане на тази помощ
    -V --version         Извеждане на версията
    -D --browse-domains  Преглед на възможните домейни, а не на услугите
    -a --all             Показване на всички услуги
    -d --domain=ДОМЕЙН   Домейнът, който да се разгледа
    -v --verbose         Включване на подробен режим
    -t --terminate       Изход след извеждането на възможно пълен списък
    -c --cache           Изход след извеждането на всички записи в кеша
    -l --ignore-local    Без локалните услуги
    -r --resolve         Откриване на адресите на намерените услуги
    -f --no-fail         Продължаване на работа дори и без демон
    -p --parsable        Изходът да може да се анализира
     -k --no-db-lookup    Без преглед на видовете услуги
    -b --dump-db         Извеждане на различните видове услуги
 %s [опции]

    -h --help            Извеждане на тази помощ
    -s --ssh             Преглед на сървърите за SSH
    -v --vnc             Преглед на сървърите за VNC
    -S --shell           Преглед на сървърите за VNC и SSH
    -d --domain=ДОМЕЙН   Домейнът, който да се разгледа
 %s [опции] %s <име на машина…>
%s [опции] %s <адрес… >

    -h --help            Извеждане на тази помощ
    -V --version         Извеждане на версията
    -n --name            Откриване на име на машина
    -a --address         Откриване на адрес
    -v --verbose         Включване на подробен режим
    -6                   Откриване на адрес по IPv6
    -4                   Откриване на адрес по IPv4
 %s [опции] %s <име> <вид> <порт> [<txt…>]
%s [fm[rr] %s <име-на-машина> <адрес>

    -h --help            Извеждане на тази помощ
    -V --version         Извеждане на версията
    -s --service         Публикуване на услуга
    -a --address         Публикуване на адрес
    -v --verbose         Включване на подробен режим
    -d --domain=ДОМЕЙН   Домейн за публикуване на услугата
    -H --host=ДОМЕЙН     Машина, на която е услугата
       --subtype=ПОДТИП  Допълнителен подтип за регистриране на услугата
    -R --no-reverse      Без публикуване на обратен запис за адреса
    -f --no-fail         Продължаване на работа дори и без демон
 %s [опции] <ново име на машината>

    -h --help            Извеждане на тази помощ
    -V --version         Извеждане на версията
    -v --verbose         Включване на подробен режим
 : всички за сега
 : кешът е изчерпан
 <i>Не е избрана услуга.</i> Списък завършващ с NULL на видовете услуги, които да се прегледат Достъпът е отказан Адрес Фамилия адреси Адрес: Неочаквана грешка от D-Bus Проблем с клиента на Avahi: %s Проблем с прегледа на домейна с Avahi: %s Проблем с откриването на Avahi: %s Неправилен брой аргументи
 Лошо състояние Преглед на видовете услуги Списъкът с услуги за преглед е празен! Неуспешен преглед за услуга от вида %s в домейна %s: %s Преглед на услугите в домейна <b>%s</b>: Преглед на услугите в <b>локалната мрежа</b>: Преглед… Отмяна.
 Смяна на домейна Избор на сървър за SSH Избор на сървър за достъп Избор на сървър за VNC Грешка в клиента, изход: %s
 Свързване към „%s“…
 Грешка от DNS: FORMERR Грешка от DNS: NOTAUTH Грешка от DNS: NOTIMP Грешка от DNS: NOTZONE Грешка от DNS: NXDOMAIN Грешка от DNS: NXRRSET Грешка от DNS: REFUSED Грешка от DNS: SERVFAIL Грешка от DNS: YXDOMAIN Грешка от DNS: YXRRSET Неуспешна връзка с демона Демонът не е стартиран С графична среда Прекъсната връзка, подновяване…
 Домейн Домейн: Сбт Инфс Пртк %-*s %-20s Домейн
 Сбт Инфс Пртк Домейн
 Установен под име „%s“
 Неуспешно добавяне на адрес: %s
 Неуспешно добавяне на услуга: %s
 Неуспешно добавяне на подтип „%s“: %s
 Неуспех при свързването със сървър на Avahi: %s Неуспешно създаване на компонент за откриване на имена: %s
 Неуспешно създаване на компонент за преглед на %s: %s Неуспех при създаване на клиентски обект: %s
 Неуспешно създаване на компонент за преглед на домейн: %s Неуспешно създаване на група записи: %s
 Неуспешно създаване на компонент за откриване на имена: %s
 Неуспешно създаване на компонент за откриване на %s от вида %s в домейна %s: %s Неуспех при създаването на прост обект за запитване.
 Адресът „%s“ не може да се анализира.
 Неуспешно разпознаване на номер на порт: %s
 Неуспешно запитване за името на машината: %s
 Неуспешно запитване за низа за версията: %s
 Проблем с прочитането на домейна с Avahi: %s Неуспешно регистриране: %s
 Неуспешно откриване на адрес „%s“: %s
 Неуспешно откриване на името „%s“: %s
 Услуга „%s“ с вид „%s“ не е открита в домейна „%s“: %s
 Име на машина Съвпадение на имена на машини
 Името на машината е променено на %s
 Инициализиране… Интерфейс: Неправилно време за живот за DNS Неправилен клас DNS Неправилен код върнат от DNS Неправилен вид DNS Неправилен код на грешка Неправилни данни RDATA Неправилен адрес Неправилен аргумент Неправилни настройки Неправилно име на домейн Неправилни флагове Неправилно име на машина Неправилен индекс на интерфейс Неправилен брой аргументи. Очаква се точно един.
 Неправилна операция Неправилен пакет Неправилен номер на порт Неправилно указан протокол Неправилен запис Неправилен ключ за запис Неправилно име на услуга Неправилен подвид на услугата Неправилен вид услуга Е празно Съвпадение на локални имена Местоположение Недостатъчно памет Име Съвпадение на имена. Избира се ново име: „%s“.
 Не е указана команда.
 Липсва подходящ мрежови протокол Не е намерено Не е позволено Не се поддържа Добре Грешка на операционната система Неуспешна операция Порт Откриване на услуга Откриване на името на машината с услугата Автоматично откриване на името на машината с избраната услуга преди връщане от функцията Автоматично откриване на избраната услуга преди връщане от функцията Ключът за запис на ресурс е шаблон Версия на сървъра: %s. Име на машината: %s
 Име на услугата Име на услугата: Вид услуга Вид услуга: TXT Данни с вид TXT Данни с вид TXT: За команден ред Портът по IP на откритата услуга Данни с вид TXT за откритата услуга Фамилията адреси за откриване на име на машина Адресът на откритата услуга Домейнът, в който да е прегледът. NULL означава стандартния домейн Името на машината на откритата услуга Подаденият обект е неправилен Заявената операция е излишна и затова неправилна Име на избраната услуга Видът на избраната услуга Времето изтече Прекалено малко аргументи
 Прекалено много аргументи
 Прекалено много клиенти Прекалено много записи Прекалено много обекти Вид Разлика във версиите Изчакване на демона…
 _Домейн… Проблем с avahi_domain_browser_new(): %s
 Проблем с avahi_service_browser_new(): %s
 Проблем с avahi_service_type_browser_new(): %s
 няма Неуспешно изпълнение на execlp(): %s
 няма Проблем със service_browser: %s
 Проблем с service_type_browser: %s
 