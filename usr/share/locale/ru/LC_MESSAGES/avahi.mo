��    �        �   
      �  �  �  h   1  �   �  i  �  b  �  �   X     �       %     5   :     p     ~     �     �  "   �     �      �     �       	   /     9  "   N  4   q  *   �  .   �        
             %     7     K     ]     z     �     �     �     �     �     �          %     ;     Q     f          �     �     �     �     �     �               9     T  %   t  &   �  #   �  #   �  #   	  !   -  (   O  <   x  %   �     �      �       #   9     ]     }  #   �  %   �  ?   �  	        )  %   =     c  
   s     ~     �     �     �     �     �     �     �          !     5     C     U  4   m     �     �     �     �     �               -     E     Z     c     x     �     �  '   �     �  &   �  	   �                 #      &      /      @      E      U   L   o   ;   �      �   "   !     :!     G!     U!     b!     p!     t!  	   }!     �!  *   �!  $   �!  +   �!  #   "  7   0"  %   h"  "   �"  4   �"  (   �"  (   #     8#     H#     [#     o#     �#     �#     �#     �#     �#  
   �#  &   �#  '   $  ,   *$     W$     ]$     r$     v$      �$    �$  �  �&  �   �*  �  *+    �,  �  �.  �   �2  ;   �3     �3  (   4  R   >4     �4  
   �4  #   �4     �4  )   �4  %   5  >   :5  >   y5  ;   �5  #   �5  "   6  ;   ;6  X   w6  0   �6  <   7     >7     L7     _7  !   {7  .   �7  !   �7  C   �7  "   28     U8     o8     �8     �8     �8     �8     �8     9     &9     A9  >   [9  "   �9     �9  V   �9  
   ,:     7:  +   V:      �:  0   �:  4   �:  6   	;  =   @;  H   ~;  U   �;  I   <  C   g<  N   �<  G   �<  Z   B=  w   �=  O   >  9   e>  8   �>  ;   �>  E   ?  7   Z?  9   �?  E   �?  J   @  n   ]@     �@  %   �@  8   A     ;A     YA     mA     �A  ,   �A     �A  $   �A  #   B     8B  !   TB  )   vB  $   �B     �B      �B  2   C  Z   5C  !   �C     �C  &   �C  <   �C     2D  &   PD  $   wD  *   �D  $   �D     �D  0   E     7E     PE     pE  D   wE  $   �E  F   �E     (F     :F  !   RF     tF     yF     �F     �F  #   �F  1   �F  }   G  m   �G  F   �G  5   7H     mH     �H     �H     �H     �H     �H     �H     �H  [   �H  @   SI  W   �I  :   �I  P   'J  ?   xJ  >   �J  l   �J  &   dK  &   �K  *   �K  -   �K  /   L  *   ;L  ,   fL  *   �L     �L  )   �L  !   �L     M  =    M  >   ^M  C   �M  
   �M  +   �M     N  $   N  )   CN     C   �   ,       '                                      r      �       �   �   T   L   |       \   %   9      e   "   �   �   H       A   �   �       �   �       �          }   j          h         +   �           #   (               �      �   �                 k   [   <   �   �       D   0   ]   x   t   w   Q   s           �   z   7   O       a   d          .           Y       {   R       5   K   y   )       /       U       c      �       !       �   �   
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
PO-Revision-Date: 2010-11-29 23:19+0000
Last-Translator: FULL NAME <EMAIL@ADDRESS>
Language-Team: Russian (http://www.transifex.com/lennart/avahi/language/ru/)
Language: ru
MIME-Version: 1.0
Content-Type: text/plain; charset=UTF-8
Content-Transfer-Encoding: 8bit
Plural-Forms: nplurals=4; plural=(n%10==1 && n%100!=11 ? 0 : n%10>=2 && n%10<=4 && (n%100<12 || n%100>14) ? 1 : n%10==0 || (n%10>=5 && n%10<=9) || (n%100>=11 && n%100<=14)? 2 : 3);
     -h --help            Вывести данную справку
    -V --version         Вывести версию
    -D --browse-domains  Искать домены вместо служб
    -a --all             Показывать все службы вне зависимости от типа
    -d --domain=DOMAIN   Искать в домене
    -v --verbose         Включить подробный режим
    -t --terminate       Показать более или менее полный список и выйти
    -c --cache           Показать все элементы в кэше и выйти
    -l --ignore-local    Игнорировать локальные службы
    -r --resolve         Преобразовывать найденные службы
    -f --no-fail         Не завершать работу, если демон недоступен
    -p --parsable        Выводить в машинном формате
     -k --no-db-lookup    Не искать типы служб
    -b --dump-db         Вывести базу типов служб
 %s [параметры]

    -h --help            Показать эту справку
    -s --ssh             Вывести список доступных серверов SSH
    -v --vnc             Вывести список доступных серверов VNC
    -S --shell           Вывести список доступных серверов SSH и VNC
    -d --domain=ДОМЕН    Домен поиска серверов
 %s [параметры] %s <имя узла...>
%s [параметры] %s <адрес... >

    -h --help            Вывести данную справку
    -V --version         Вывести версию
    -n --name            Преобразовать имя узла
    -a --address         Преобразовать адрес
    -v --verbose         Включить подробный режим
    -6                   Искать адрес IPv6
    -4                   Искать адрес IPv4
 %s [параметры] %s <имя> <тип> <порт> [<текст...>]
%s [параметры] %s <имя узла> <адрес>

    -h --help            Вывести эту справку
    -V --version         Вывести версию
    -s --service         Опубликовать службу
    -a --address         Опубликовать адрес
    -v --verbose         Включить подробный режим
    -d --domain=ДОМЕН    Домен для публикации службы
    -H --host=ДОМЕН      Узел, на котором размещается служба
       --subtype=ПОДТИП  Дополнительный подтип для регистрации этой службы
    -R --no-reverse      Не опубликовывать элемент обратного адреса
    -f --no-fail         Не завершать работу, если процесс службы недоступен
 %s [параметры] <новое имя узла>

    -h --help            Вывести данную справку
    -V --version         Вывести версию
    -v --verbose         Включить подробный режим
 : Больше ничего на данный момент
 : Кэш исчерпан
 <i>Служба не выбрана.</i> Список типов искомых служб, завершающийся NULL Доступ запрещён Адрес Адресное семейство Адрес: Неожиданная ошибка D-Bus Ошибка клиента Avahi: %s Ошибка обозревателя доменов Avahi: %s Ошибка преобразователя имён Avahi: %s Неверное количество аргументов
 Неверное состояние Типы искомых служб Список типов искомых служб пуст! Не удалось выполнить поиск службы %s в домене %s: %s Поиск служб в домене <b>%s</b>: Поиск служб в <b>локальной сети</b>: Поиск... Отменено.
 Изменить домен Выберите SSH-сервер Выберите сервер оболочки Выберите VNC-сервер Ошибка клиента, завершение работы: %s
 Соединение с «%s»...
 Ошибка DNS: FORMERR Ошибка DNS: NOTAUTH Ошибка DNS: NOTIMP Ошибка DNS: NOTZONE Ошибка DNS: NXDOMAIN Ошибка DNS: NXRRSET Ошибка DNS: REFUSED Ошибка DNS: SERVFAIL Ошибка DNS: YXDOMAIN Ошибка DNS: YXRRSET Не удалось соединиться со службой Служба не запущена Рабочий стол Соединение разорвано, повторное подключение...
 Домен Название домена: А Интр Прот %-*s %-20s Домен
 А Интр Прот Домен
 Образована под именем «%s»
 Не удалось добавить адрес: %s
 Не удалось добавить службу: %s
 Не удалось добавить подтип «%s»: %s
 Не удалось подключиться к серверу Avahi: %s Не удалось создать преобразователь адресов: %s
 Не удалось создать обозреватель для %s: %s Не удалось создать объект клиента: %s
 Не удалось создать обозреватель доменов: %s Не удалось создать группу элементов: %s
 Не удалось создать преобразователь имён узлов: %s
 Не удалось создать преобразователь имён для %s типа %s в домене %s: %s Не удалось создать объект простого опроса.
 Не удалось разобрать адрес «%s»
 Ошибка разбора номера порта: %s
 Не удалось запросить имя узла: %s
 Не удалось запросить строку версии: %s
 Не удалось открыть домен Avahi: %s Не удалось зарегистрировать: %s
 Не удалось преобразовать адрес «%s»: %s
 Не удалось преобразовать имя узла «%s»: %s
 Не удалось преобразовать службу «%s» типа «%s» в домене «%s»: %s
 Имя узла Конфликт имени узла
 Имя узла успешно изменено на %s
 Инициализация... Интерфейс: Неверный TTL DNS Неверный класс DNS Неверный код возврата DNS Неверный тип DNS Неверный код ошибки Неверный формат RDATA Неверный адрес Неверный аргумент Неверная конфигурация Неверное имя домена Неверные флаги Неверное имя узла Неверный индекс интерфейса Неверное число аргументов, ожидается ровно один.
 Неверная операция Неверный пакет Неверный номер порта Неверная спецификация протокола Неверная запись Неверный ключ записи Неверное имя службы Неверный подтип службы Неверный тип службы Пустая группа Перекрытие локальных имён Расположение Память исчерпана Имя Коллизия имён, выбрано новое имя «%s».
 Не указана команда.
 Не найден подходящий сетевой протокол Не найден Не разрешено Не поддерживается ОК Ошибка ОС Сбой операции Порт Определение службы Определить имя узла службы Автоматически определить имя узла выбранной службы перед возвратом Автоматически определить выбранную службу перед возвратом Ключ записи ресурса является шаблоном Версия сервера: %s; Имя узла: %s
 Имя службы Имя службы: Тип службы Тип службы: TXT Данные TXT Данные TXT: Терминал Номер порта протокола IP определённого устройства Данные TXT определённого устройства Определение адресного семейства для имени узла Адрес определённого устройства Домен поиска или домен по умолчанию, если NULL Имя узла определённого устройства Переданный объект недействителен Запрошенная операция неверна, поскольку является излишней Имя выбранной службы Тип выбранной службы Время ожидания истекло Слишком мало аргументов
 Слишком много аргументов
 Слишком много клиентов Слишком много элементов Слишком много объектов Тип Несоответствие версий Ожидание демона...
 _Домен... Сбой выполнения avahi_domain_browser_new(): %s
 Сбой выполнения avahi_service_browser_new(): %s
 Сбой выполнения avahi_service_type_browser_new(): %s
 пусто Сбой выполнения execlp(): %s
 н/д Ошибка в service_browser: %s
 Ошибка в service_type_browser: %s
 