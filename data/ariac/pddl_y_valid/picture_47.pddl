(define (problem picture_47)
    (:domain ariac)
    
    (:objects
        table pump_placement regulator_placement battery_placement buffer_placement - location
        red_pump red_sensor red_battery - part
    )
    
    (:init
        (robot_at table)
        (handempty)        
        (clear red_pump)
        (clear red_sensor)
        (clear red_battery)
        (part_at red_pump table)
        (part_at red_sensor table)
        (part_at red_battery table)
    )
    
    (:goal
        (and
            (part_at red_pump pump_placement)
            (part_at red_battery battery_placement)
        )
    )
)
