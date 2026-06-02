(define (problem picture_37)
    (:domain ariac)
    
    (:objects
        table pump_placement regulator_placement battery_placement buffer_placement - location
        red_sensor red_regulator green_battery - part
    )
    
    (:init
        (robot_at table)
        (handempty)        
        (clear red_sensor)
        (clear red_regulator)
        (clear green_battery)
        (part_at red_regulator table)
        (part_at red_sensor table)
        (part_at green_battery table)
    )
    
    (:goal
        (and
            (part_at red_regulator regulator_placement)
            (part_at green_battery battery_placement)
        )
    )
)
